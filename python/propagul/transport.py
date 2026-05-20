"""GossipTransport — asyncio TCP-based gossip networking layer.

Protocol: length-prefixed frames with optional HMAC authentication.

    Without auth:
        | 4 bytes length (big-endian u32) | JSON payload |

    With auth (gossip_secret set):
        | 4 bytes length (big-endian u32) | 32 bytes HMAC-SHA256 | JSON payload |

    The length field always covers the entire frame body (HMAC + payload).

Each node runs a TCP server and connects to peers as a client.
The gossip scheduler controls how many peers (k) receive state each round.

Connection pooling: outbound connections are kept alive and reused
across gossip rounds. This amortizes the TLS handshake cost (which is
~2ms per connection) over many messages. Stale connections are evicted
after POOL_TTL_SECONDS of inactivity.
"""

import asyncio
import hashlib
import hmac as hmac_mod
import ssl
import struct
import time
import logging
import random
from typing import Callable, Optional, List, Set, Dict, Tuple

from propagul.types import PeerAddress

logger = logging.getLogger("propagul.transport")

# Protocol constants
HEADER_SIZE = 4  # 4 bytes for u32 length prefix
HMAC_SIZE = 32  # SHA-256 HMAC
MAX_MESSAGE_SIZE = 1_048_576  # 1 MB (matches Rust-side limit)
CONNECT_TIMEOUT = 5.0  # seconds
SEND_TIMEOUT = 3.0  # seconds

# Connection pool settings
POOL_MAX_SIZE = 32
POOL_TTL_SECONDS = 30

# Inbound connection limit — prevents FD exhaustion DoS (P2-01)
MAX_INBOUND_CONNECTIONS = 128
INBOUND_IDLE_TIMEOUT = 60.0  # seconds

# Eviction throttle — only run every N send rounds
EVICT_INTERVAL_ROUNDS = 10


class _PooledConnection:
    """A cached outbound TCP connection with TTL tracking and write lock."""

    __slots__ = ("reader", "writer", "last_used", "peer", "lock")

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer: PeerAddress,
    ):
        self.reader = reader
        self.writer = writer
        self.last_used = time.monotonic()
        self.peer = peer
        self.lock = asyncio.Lock()  # Serializes writes to prevent interleaving

    @property
    def is_alive(self) -> bool:
        """Check if connection is still usable."""
        if self.writer.is_closing():
            return False
        if self.reader.at_eof():
            return False
        if time.monotonic() - self.last_used > POOL_TTL_SECONDS:
            return False
        return True

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def close(self) -> None:
        """Close connection and await cleanup."""
        try:
            self.writer.close()
            await asyncio.wait_for(self.writer.wait_closed(), timeout=1.0)
        except Exception:
            pass


class GossipTransport:
    """Asyncio TCP gossip transport with optional HMAC authentication.

    Runs a TCP server and sends gossip messages to peers.
    All operations are single-loop asyncio (NOT thread-safe
    across event loops — use from one loop only).

    Outbound connections are pooled: a TCP+TLS connection is opened once
    per peer and reused for subsequent messages. Idle connections are
    evicted after POOL_TTL_SECONDS (30s default).

    If gossip_secret is provided, all messages are authenticated with
    HMAC-SHA256. Messages without valid HMAC are silently dropped.
    This prevents unauthorized peers from injecting state.

    Usage:
        transport = GossipTransport(
            bind_address=PeerAddress("0.0.0.0", 9001),
            on_receive=handle_incoming_state,
            gossip_secret=b"my-room-secret",  # Optional
        )
        await transport.start()
        delivered = await transport.send(peer, snapshot_bytes)
        await transport.stop()
    """

    def __init__(
        self,
        bind_address: PeerAddress,
        on_receive: Callable[[bytes, PeerAddress], None],
        gossip_secret: Optional[bytes] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        insecure_tls: bool = False,
    ):
        self._bind = bind_address
        self._on_receive = on_receive
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._active_connections: Set[asyncio.StreamWriter] = set()
        self._secret = gossip_secret

        # TLS: server context for accepting connections
        self._server_ssl = ssl_context

        # TLS: client context for outbound connections
        self._client_ssl: Optional[ssl.SSLContext] = None
        if ssl_context is not None:
            if insecure_tls:
                logger.warning("TLS client verification DISABLED (insecure_tls=True)")
                self._client_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                self._client_ssl.check_hostname = False
                self._client_ssl.verify_mode = ssl.CERT_NONE
            else:
                # Production: Use provided context (which may be configured for mTLS or custom CA)
                # Fallback to default if they didn't configure client-side properly, but honoring
                # the provided context is required if they passed it to support custom CAs.
                self._client_ssl = ssl_context
            self._client_ssl.minimum_version = ssl.TLSVersion.TLSv1_2

        # Connection pool: peer → pooled connection
        self._pool: Dict[PeerAddress, _PooledConnection] = {}
        self._pool_lock: Optional[asyncio.Lock] = None  # Created lazily in start()

        # Metrics
        self.bytes_sent: int = 0
        self.bytes_received: int = 0
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.send_errors: int = 0
        self.auth_rejections: int = 0
        self.pool_hits: int = 0
        self.pool_misses: int = 0
        self._send_rounds: int = 0  # Counter for eviction throttle
        self._rejected_connections: int = 0  # P2-01 metric

    async def start(self) -> None:
        """Start the TCP server."""
        if self._running:
            return
        # Create lock inside event loop (required for Python 3.9)
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._bind.host,
            self._bind.port,
            reuse_address=True,
            ssl=self._server_ssl,
        )
        self._running = True
        tls_mode = "TLS" if self._server_ssl else "plain"
        auth_mode = "HMAC" if self._secret else "none"
        logger.info("Gossip server listening on %s (%s, auth=%s)", self._bind, tls_mode, auth_mode)

    async def stop(self) -> None:
        """Stop the TCP server gracefully with timeout.

        Forces close of lingering connections after 2s to prevent
        shutdown deadlocks from stale peer connections.
        """
        self._running = False

        # Close all pooled outbound connections
        for conn in self._pool.values():
            await conn.close()
        self._pool.clear()

        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Transport wait_closed timed out, forcing shutdown")
            self._server = None

        # Force-close any lingering inbound connections
        for writer in list(self._active_connections):
            try:
                writer.close()
            except Exception:
                pass
        self._active_connections.clear()
        logger.info("Gossip server stopped")

    def _compute_hmac(self, payload: bytes) -> bytes:
        """Compute HMAC-SHA256 of the payload."""
        return hmac_mod.new(self._secret, payload, hashlib.sha256).digest()

    async def _get_connection(self, peer: PeerAddress) -> Optional[_PooledConnection]:
        """Get a pooled connection or create a new one.

        Returns None if connection cannot be established.
        All pool mutations are guarded by _pool_lock.
        """
        conn_to_close = None
        async with self._pool_lock:
            # Try pool first
            cached = self._pool.get(peer)
            if cached is not None and cached.is_alive:
                self.pool_hits += 1
                cached.touch()
                return cached

            # Evict stale entry
            if cached is not None:
                conn_to_close = self._pool.pop(peer)

        if conn_to_close:
            await conn_to_close.close()

        # Create new connection OUTSIDE the lock (blocking IO)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port, ssl=self._client_ssl),
                timeout=CONNECT_TIMEOUT,
            )
        except (ConnectionRefusedError, ConnectionResetError, OSError, asyncio.TimeoutError) as e:
            logger.debug("Connection to %s failed: %s", peer, e)
            return None

        conn = _PooledConnection(reader, writer, peer)
        self.pool_misses += 1

        conn_to_evict = None
        async with self._pool_lock:
            # Evict oldest if pool is full
            if len(self._pool) >= POOL_MAX_SIZE:
                oldest_peer = min(self._pool, key=lambda p: self._pool[p].last_used)
                conn_to_evict = self._pool.pop(oldest_peer)

            self._pool[peer] = conn
            
        if conn_to_evict:
            await conn_to_evict.close()
            
        return conn

    async def _evict_stale(self) -> None:
        """Remove stale connections from the pool."""
        conns_to_close = []
        async with self._pool_lock:
            stale = [p for p, c in self._pool.items() if not c.is_alive]
            for p in stale:
                conns_to_close.append(self._pool.pop(p))
                
        for c in conns_to_close:
            await c.close()

    async def send(self, peer: PeerAddress, data: bytes) -> bool:
        """Send a gossip message to a peer. Returns True if delivered.

        Uses connection pooling to reuse TCP+TLS connections across rounds.
        If a pooled connection fails (broken pipe), it is evicted and a
        fresh connection is established for retry.
        """
        if not self._running:
            return False
        if self._pool_lock is None:
            logger.error("Cannot send gossip message: GossipTransport is not started (pool_lock is None)")
            return False  # Not started yet

        # Pre-flight size check — reject before opening connection
        frame_size = len(data) + (HMAC_SIZE if self._secret else 0)
        if frame_size > MAX_MESSAGE_SIZE:
            logger.warning("Rejecting oversized outbound message: %d bytes", frame_size)
            return False

        # Build frame once
        if self._secret:
            mac = self._compute_hmac(data)
            frame_body = mac + data
        else:
            frame_body = data
        frame = struct.pack(">I", len(frame_body)) + frame_body

        # Try with pooled connection, retry once on broken pipe
        for attempt in range(2):
            conn = await self._get_connection(peer)
            if conn is None:
                self.send_errors += 1
                return False

            try:
                async with conn.lock:  # Serialize writes on this connection
                    conn.writer.write(frame)
                    await asyncio.wait_for(conn.writer.drain(), timeout=SEND_TIMEOUT)
                conn.touch()  # Update last_used after successful send
                self.bytes_sent += len(data)
                self.messages_sent += 1
                return True
            except (ConnectionResetError, BrokenPipeError, OSError, asyncio.TimeoutError) as e:
                logger.debug("Send to %s failed (attempt %d): %s", peer, attempt + 1, e)
                # Evict broken connection under lock and retry
                await conn.close()
                async with self._pool_lock:
                    self._pool.pop(peer, None)
                if attempt == 0:
                    continue  # Retry with fresh connection
                self.send_errors += 1
                return False

        self.send_errors += 1
        return False

    async def send_to_k_peers(
        self,
        peers: List[PeerAddress],
        data: bytes,
        k: int,
    ) -> List[bool]:
        """Send gossip to k randomly-selected peers from the list.

        Returns a list of delivery results (True/False) for each selected peer.
        Periodically evicts stale pool entries.
        """
        if k <= 0 or not peers:
            return []

        # Lazy pool maintenance — throttled to every N rounds (P1-evict)
        self._send_rounds += 1
        if self._send_rounds % EVICT_INTERVAL_ROUNDS == 0:
            await self._evict_stale()

        selected = random.sample(peers, min(k, len(peers)))
        tasks = [self.send(peer, data) for peer in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        delivery_results = []
        for result in results:
            if isinstance(result, Exception):
                delivery_results.append(False)
            else:
                delivery_results.append(bool(result))
        return delivery_results

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming gossip connection.

        Reads multiple frames per connection (persistent connection support).
        The connection stays open until the peer closes it, a timeout occurs,
        or an error is encountered.
        """
        # P2-01: Reject if too many concurrent connections
        if len(self._active_connections) >= MAX_INBOUND_CONNECTIONS:
            self._rejected_connections += 1
            logger.warning(
                "Connection limit reached (%d), rejecting %s",
                MAX_INBOUND_CONNECTIONS, writer.get_extra_info("peername"),
            )
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except Exception:
                pass
            return

        self._active_connections.add(writer)
        peer_info = writer.get_extra_info("peername")
        try:
            while self._running:
                # Read length header
                try:
                    header = await asyncio.wait_for(
                        reader.readexactly(HEADER_SIZE),
                        timeout=INBOUND_IDLE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    break  # Idle timeout — close gracefully
                except asyncio.IncompleteReadError:
                    break  # Peer closed connection

                (length,) = struct.unpack(">I", header)

                if length > MAX_MESSAGE_SIZE:
                    logger.warning(
                        "Rejected oversized message from %s: %d bytes", peer_info, length
                    )
                    break

                # Read frame body
                frame_body = await asyncio.wait_for(
                    reader.readexactly(length),
                    timeout=SEND_TIMEOUT,
                )

                # Extract payload (with or without HMAC verification)
                if self._secret:
                    if length < HMAC_SIZE:
                        logger.warning("Rejected short HMAC frame from %s", peer_info)
                        self.auth_rejections += 1
                        break

                    received_mac = frame_body[:HMAC_SIZE]
                    data = frame_body[HMAC_SIZE:]

                    expected_mac = self._compute_hmac(data)
                    if not hmac_mod.compare_digest(received_mac, expected_mac):
                        logger.warning("Rejected invalid HMAC from %s", peer_info)
                        self.auth_rejections += 1
                        break
                else:
                    data = frame_body

                self.bytes_received += len(data)
                self.messages_received += 1

                # Deliver to application (wrapped to prevent callback crash
                # from killing the transport connection — P0-03)
                peer_addr = PeerAddress(
                    host=peer_info[0] if peer_info else "unknown",
                    port=peer_info[1] if peer_info else 0,
                )
                try:
                    self._on_receive(data, peer_addr)
                except Exception as cb_err:
                    logger.error("on_receive callback error: %s", cb_err)

        except Exception as e:
            logger.debug("Connection error from %s: %s", peer_info, e)
        finally:
            self._active_connections.discard(writer)
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pool_size(self) -> int:
        """Number of active pooled outbound connections."""
        return len(self._pool)

    def set_client_tls(self, ctx: ssl.SSLContext) -> None:
        """Set the client-side TLS context for outbound connections.

        Used by connect_to_cloud() to auto-enable TLS when the cloud
        service reports TLS support. (P2-04: replaces private access)
        """
        self._client_ssl = ctx


def create_tls_context(
    certfile: str,
    keyfile: str,
    ca_certfile: Optional[str] = None,
) -> ssl.SSLContext:
    """Create a TLS context for the gossip transport.

    For the cloud peer: use Let's Encrypt certs.
    For local dev: use self-signed certs or skip TLS.

    Args:
        certfile: Path to PEM certificate file.
        keyfile: Path to PEM private key file.
        ca_certfile: Optional CA certificate for mutual TLS.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    if ca_certfile:
        ctx.load_verify_locations(ca_certfile)
        ctx.verify_mode = ssl.CERT_REQUIRED
    else:
        ctx.verify_mode = ssl.CERT_NONE
    # Disable old protocols
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx
