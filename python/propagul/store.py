"""AgentStateStore — High-level API for crash-resilient agent state sync.

This is the primary user-facing class. It combines a pure Python OR-Map CRDT
with push-pull gossip-based synchronization.

Usage:
    store = AgentStateStore(room="my-project", node_id=1, port=9001)
    store.set("task", "research")
    store.set("status", "running")

    async with store:
        # Gossip loop runs in background
        await asyncio.sleep(60)

    # Or manually:
    await store.start(peers=[PeerAddress("127.0.0.1", 9002)])
    ...
    await store.stop()
"""

import asyncio
import time
import json
import logging
import threading
from typing import Optional, List, Callable, Dict, Protocol, runtime_checkable

from propagul.crdt import ORMap
from propagul.gossip import (
    GossipStats,
    build_full_state_message,
    build_digest_message,
    handle_incoming_message,
    compute_jitter,
    DEFAULT_INTERVAL_MS,
    DEFAULT_K,
)
from propagul.types import (
    PeerAddress,
    StoreStats,
    ChangeCallback,
)
from propagul.transport import GossipTransport
import ssl as _ssl_mod
from propagul.discovery import StaticDiscovery

logger = logging.getLogger("propagul.store")



@runtime_checkable
class GossipScheduler(Protocol):
    """Protocol for adaptive gossip scheduling.

    The Cloud Peer provides an EntropyAgent that implements this.
    SDK users don't need this — fixed k=1 is the default.
    """

    def get_k(self, current_time_ms: float, state_total: int) -> int: ...
    def on_packet_sent(self, delivered: bool) -> None: ...
    def on_receive(self, delta: int, current_time_ms: float) -> None: ...


class AgentStateStore:
    """AgentStateStore — crash-resilient state with push-pull gossip sync.

    Provides a simple key-value interface backed by a conflict-free
    replicated data type (OR-Map CRDT). State is synchronized with
    peers via TCP gossip using push-pull exchange.

    Free (no cloud): Fixed k=1 push gossip. Works fully P2P.
    With cloud peer: Enhanced scheduling, persistence, always-online.

    Args:
        room: Logical room/namespace for this state group.
        node_id: Unique identifier for this node (must be unique per room).
        port: TCP port to listen on for gossip.
        host: TCP host to bind to (default: 0.0.0.0).
        gossip_interval_ms: Base gossip interval in milliseconds.
        gossip_secret: Optional HMAC secret for authenticated gossip.
        ssl_context: Optional ssl.SSLContext for TLS-encrypted gossip.
        gossip_scheduler: Optional adaptive scheduler (e.g., EntropyAgent).
            If not provided, uses fixed k=1 (SDK default).
    """

    def __init__(
        self,
        room: str,
        node_id: int,
        port: int = 9000,
        host: str = "0.0.0.0",
        gossip_interval_ms: float = DEFAULT_INTERVAL_MS,
        gossip_secret: Optional[bytes] = None,
        ssl_context: Optional[_ssl_mod.SSLContext] = None,
        gossip_scheduler: Optional[GossipScheduler] = None,
    ):
        self.room = room
        self.node_id = node_id
        self._gossip_interval_ms = gossip_interval_ms

        # State: Pure Python OR-Map CRDT
        self._crdt = ORMap(node_id)
        self._crdt_lock = threading.Lock()

        # Gossip stats
        self._gossip_stats = GossipStats()

        # Networking
        self._bind = PeerAddress(host, port)
        self._transport = GossipTransport(
            bind_address=self._bind,
            on_receive=self._on_gossip_receive,
            gossip_secret=gossip_secret,
            ssl_context=ssl_context,
        )
        self._discovery = StaticDiscovery()
        self._scheduler = gossip_scheduler  # None = fixed k=1

        # State tracking
        self._running = False
        self._gossip_task: Optional[asyncio.Task] = None
        self._change_callbacks: List[ChangeCallback] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ─── Public API ──────────────────────────────────────────────────────

    def set(self, key: str, value: str) -> None:
        """Set a key to a value (replaces previous)."""
        with self._crdt_lock:
            self._crdt.set(key, value)

    def get(self, key: str) -> Optional[str]:
        """Get the current value for a key."""
        with self._crdt_lock:
            return self._crdt.get(key)

    def get_all(self) -> Dict[str, str]:
        """Get all key-value pairs."""
        with self._crdt_lock:
            result = {}
            for key in self._crdt.keys():
                val = self._crdt.get(key)
                if val is not None:
                    result[key] = val
            return result

    def delete(self, key: str) -> None:
        """Delete a key (propagates via tombstone)."""
        with self._crdt_lock:
            self._crdt.delete(key)

    def get_conflicts(self, key: str) -> List[str]:
        """Get all concurrent values for a key (conflict detection)."""
        with self._crdt_lock:
            return self._crdt.get_all(key)

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a callback for remote state changes.

        callback(key: str, new_value: Optional[str]) is called when
        a remote peer changes a key via gossip merge.
        """
        self._change_callbacks.append(callback)

    @property
    def stats(self) -> StoreStats:
        """Current telemetry snapshot."""
        return StoreStats(
            entropy=0.0,
            current_k=DEFAULT_K,
            sleep_ratio=0.0,
            loss_rate=0.0,
            convergence_ms=0.0,
            is_partitioned=False,
            shock_events=0,
            key_count=len(self._crdt),
            tombstone_count=self._crdt.tombstone_count,
            peer_count=self._discovery.peer_count,
            gossip_rounds=self._gossip_stats.rounds,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    # ─── Lifecycle ───────────────────────────────────────────────────────

    async def start(
        self,
        peers: Optional[List[PeerAddress]] = None,
    ) -> None:
        """Start the gossip transport and gossip loop.

        Args:
            peers: Initial peer list. Can also be added later via add_peer().
        """
        if self._running:
            return

        self._loop = asyncio.get_event_loop()

        # Register initial peers
        if peers:
            for peer in peers:
                # Accept both PeerAddress and raw (host, port) tuples
                if isinstance(peer, (tuple, list)) and len(peer) == 2:
                    peer = PeerAddress(str(peer[0]), int(peer[1]))
                self._discovery.add_peer(peer)

        # Start TCP server
        await self._transport.start()
        self._running = True

        # Anti-entropy sync: push our state AND request theirs.
        # 1. Send FULL_STATE so peers get our data immediately
        # 2. Send DIGEST so peers respond with their delta (pull)
        peers_list = self._discovery.get_peers(exclude=self._bind)
        if peers_list:
            push_msg = build_full_state_message(self._crdt, reply_port=self._bind.port)
            pull_msg = build_digest_message(self._crdt, reply_port=self._bind.port)
            for peer in peers_list:
                await self._transport.send(peer, push_msg)
                await self._transport.send(peer, pull_msg)

        # Start gossip loop
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(
            "AgentStateStore started: room=%s node=%d port=%d peers=%d",
            self.room,
            self.node_id,
            self._bind.port,
            self._discovery.peer_count,
        )

    async def stop(self) -> None:
        """Stop gossip and transport.

        Uses timeouts to prevent shutdown deadlocks from lingering
        TCP connections or in-flight gossip sends.
        """
        self._running = False
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._gossip_task), timeout=3.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._gossip_task = None
        await self._transport.stop()
        logger.info("AgentStateStore stopped: room=%s", self.room)

    async def __aenter__(self) -> "AgentStateStore":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    # ─── Peer Management ─────────────────────────────────────────────────

    def add_peer(self, peer: PeerAddress) -> None:
        """Add a peer for gossip."""
        self._discovery.add_peer(peer)

    def remove_peer(self, peer: PeerAddress) -> None:
        """Remove a peer from gossip."""
        self._discovery.remove_peer(peer)

    def connect_to_cloud(
        self,
        api_key: str,
        cloud_host: str = "cloud.propagul.dev",
        cloud_api_port: int = 443,
    ) -> None:
        """Connect to the Propagul cloud peer.

        Registers a room on the managed service and adds the cloud peer
        for persistent state synchronization. Automatically enables TLS
        for gossip transport if the cloud service reports TLS support.

        The cloud peer keeps your state alive even when all your agents
        are offline. When an agent restarts, it syncs from the cloud peer.

        This is fully self-service: the SDK handles TLS negotiation,
        API authentication, and peer discovery automatically.

        Note: If called from within an async event loop, the blocking HTTP
        request is offloaded to a thread executor (P1-03) to avoid stalling
        the gossip loop.

        Args:
            api_key: Your Propagul API key (e.g., "pg_live_abc123").
            cloud_host: Cloud service hostname.
            cloud_api_port: Cloud service HTTPS API port (default: 443).

        Raises:
            RuntimeError: If the cloud service is unreachable or returns an error.
        """
        import ssl as _ssl
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        ssl_ctx = _ssl.create_default_context()

        url = f"https://{cloud_host}:{cloud_api_port}/rooms"
        payload = json.dumps({"room_id": self.room}).encode()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def _register_peer(resp_data: dict) -> None:
            gossip_host = resp_data.get("gossip_host", cloud_host)
            gossip_port = resp_data.get("gossip_port")
            gossip_tls = resp_data.get("gossip_tls", True)  # Default: TLS on

            if gossip_port:
                cloud_peer = PeerAddress(gossip_host, int(gossip_port))
                self._discovery.add_peer(cloud_peer)

                # P2-04: Use public setter instead of private access
                if gossip_tls and self._transport._client_ssl is None:
                    # Production: verify server cert via system CA bundle
                    client_ssl = _ssl.create_default_context()
                    client_ssl.minimum_version = _ssl.TLSVersion.TLSv1_2
                    self._transport.set_client_tls(client_ssl)
                    logger.info("Auto-enabled TLS for cloud gossip (cert verification ON)")

                logger.info(
                    "Connected to cloud peer: %s:%d (room=%s, tier=%s, tls=%s)",
                    gossip_host, gossip_port, self.room,
                    resp_data.get("tier"), gossip_tls,
                )

        def _do_http_registration() -> None:
            """Blocking HTTP registration — runs in executor if async."""
            req = Request(url, data=payload, headers=headers, method="POST")
            try:
                with urlopen(req, timeout=10, context=ssl_ctx) as resp:
                    _register_peer(json.loads(resp.read()))
            except HTTPError as e:
                if e.code == 409:
                    # Room already exists — get its info
                    info_url = f"https://{cloud_host}:{cloud_api_port}/rooms/{self.room}"
                    info_req = Request(info_url, headers=headers)
                    try:
                        with urlopen(info_req, timeout=10, context=ssl_ctx) as resp2:
                            _register_peer(json.loads(resp2.read()))
                    except Exception as e2:
                        raise RuntimeError(f"Failed to reconnect to cloud room: {e2}") from e2
                else:
                    body = e.read().decode() if hasattr(e, 'read') else str(e)
                    raise RuntimeError(f"Cloud API error ({e.code}): {body}") from e
            except URLError as e:
                raise RuntimeError(
                    f"Cannot reach cloud service at {cloud_host}:{cloud_api_port}: {e}"
                ) from e

        # P1-03: If running inside an event loop, offload to executor
        # to avoid blocking the gossip loop. Otherwise, run synchronously.
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context — schedule in thread pool
            import concurrent.futures
            future = loop.run_in_executor(None, _do_http_registration)
            # Fire-and-forget with error logging
            task = asyncio.ensure_future(future)
            task.add_done_callback(self._log_send_error)
            logger.debug("Cloud registration scheduled in executor (non-blocking)")
        except RuntimeError:
            # No running event loop — safe to block
            _do_http_registration()

    # ─── Internal: Gossip Loop ────────────────────────────────────────────

    async def _gossip_loop(self) -> None:
        """Main gossip loop — runs until stopped.

        Fixed k=1, push-based gossip with jitter.
        Each round sends a FULL_STATE message to one random peer.

        For push-pull: incoming DIGEST messages trigger delta responses
        via _on_gossip_receive → handle_incoming_message.
        """
        while self._running:
            try:
                peers = self._discovery.get_peers(exclude=self._bind)
                self._gossip_stats.rounds += 1

                with self._crdt_lock:
                    has_data = len(self._crdt) > 0
                if peers and has_data:
                    # Determine k: adaptive (if scheduler) or fixed k=1
                    if self._scheduler is not None:
                        now_ms = time.monotonic() * 1000.0
                        with self._crdt_lock:
                            state_total = len(self._crdt) + self._crdt.tombstone_count
                        k = self._scheduler.get_k(now_ms, state_total)
                    else:
                        k = DEFAULT_K

                    if k > 0:
                        # Push: send full state to k random peers
                        with self._crdt_lock:
                            msg = build_full_state_message(self._crdt, reply_port=self._bind.port)
                        results = await self._transport.send_to_k_peers(
                            peers, msg, k
                        )
                        self._gossip_stats.push_sent += 1

                        # Report delivery results to scheduler
                        if self._scheduler is not None:
                            for delivered in results:
                                self._scheduler.on_packet_sent(delivered)

                # Sleep with jitter to prevent thundering herd
                interval = compute_jitter(self._gossip_interval_ms) / 1000.0
                await asyncio.sleep(max(0.05, interval))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Gossip loop error: %s", e, exc_info=True)
                await asyncio.sleep(1.0)

    def _on_gossip_receive(self, data: bytes, sender: PeerAddress) -> None:
        """Handle incoming gossip message (called by transport).

        Delegates to gossip.handle_incoming_message which handles
        FULL_STATE, DIGEST, and DELTA message types.

        If a response is needed (for DIGEST pull-requests), it is
        sent back asynchronously.
        """
        if not self._running:
            return

        try:
            # P1-04: Only snapshot pre-state if change callbacks are registered
            has_callbacks = bool(self._change_callbacks)
            
            with self._crdt_lock:
                if has_callbacks:
                    pre_keys = set(self._crdt.keys())
                    pre_values: Dict[str, Optional[str]] = {}
                    for key in pre_keys:
                        pre_values[key] = self._crdt.get(key)

                # Process message (may merge state, may generate response)
                response, sender_listen_port = handle_incoming_message(
                    self._crdt, data, self._gossip_stats
                )

                if has_callbacks:
                    post_keys = set(self._crdt.keys())
                    post_values: Dict[str, Optional[str]] = {}
                    for key in post_keys:
                        post_values[key] = self._crdt.get(key)

            # Auto-discover sender: register their listening port
            if sender_listen_port > 0:
                real_sender = PeerAddress(sender.host, sender_listen_port)
                if real_sender != self._bind:
                    self._discovery.add_peer(real_sender)

            # P1-02: If a response is needed, schedule with error logging
            if response is not None and sender_listen_port > 0:
                real_sender = PeerAddress(sender.host, sender_listen_port)
                if self._loop is not None:
                    # _on_gossip_receive runs in the event loop thread, so create_task is safe
                    task = self._loop.create_task(
                        self._transport.send(real_sender, response)
                    )
                    task.add_done_callback(self._log_send_error)

            # Fire change callbacks (only if pre-snapshot was taken)
            if has_callbacks:
                # Detect added + modified keys
                changed_keys: set = set()
                for key in post_keys:
                    new_val = post_values.get(key)
                    old_val = pre_values.get(key)
                    if new_val != old_val:
                        changed_keys.add(key)

                # Detect deleted keys (tombstone propagation)
                deleted_keys = pre_keys - post_keys

                if changed_keys or deleted_keys:
                    for key in changed_keys:
                        new_val = post_values.get(key)
                        for cb in self._change_callbacks:
                            try:
                                cb(key, new_val)
                            except Exception as e:
                                logger.error("Change callback error: %s", e)

                    for key in deleted_keys:
                        for cb in self._change_callbacks:
                            try:
                                cb(key, None)
                            except Exception as e:
                                logger.error("Change callback error: %s", e)

        except Exception as e:
            logger.error("Gossip receive error from %s: %s", sender, e)

    @staticmethod
    def _log_send_error(task: asyncio.Task) -> None:
        """Done-callback for gossip response tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug("Gossip response send failed: %s", exc)
