"""Push-Pull Gossip Protocol.

Implements efficient state synchronization using digest-based exchange:
1. PUSH: Send our full snapshot to k peers (for fast initial sync)
2. PULL: Exchange digests, then send only missing deltas

Message types:
- FULL_STATE: Complete snapshot (used for push and recovery)
- DIGEST: Key→version mapping (used for pull negotiation)
- DELTA: Partial snapshot (only missing entries + tombstones)

Wire format: JSON with "type" field discriminator.
"""

from __future__ import annotations

import json
import hashlib
import random
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from propagul.crdt import ORMap

logger = logging.getLogger("propagul.gossip")

# Message type constants
MSG_FULL_STATE = "full_state"
MSG_DIGEST = "digest"
MSG_DELTA = "delta"
MSG_DIGEST_RESPONSE = "digest_response"

# Gossip defaults
DEFAULT_INTERVAL_MS = 500.0
DEFAULT_K = 1
JITTER_MS = 100.0

# P2-03: Max message size before JSON parse (10 MB)
MAX_MESSAGE_BYTES = 10 * 1024 * 1024


@dataclass
class GossipStats:
    """Gossip loop statistics."""
    rounds: int = 0
    push_sent: int = 0
    pull_sent: int = 0
    merges: int = 0
    delta_keys_received: int = 0


def build_digest(crdt: ORMap) -> Dict[str, Tuple[int, int]]:
    """Build a digest: key → highest tag for that key.

    Used to compare state versions without sending full data.
    """
    digest = {}
    for key in crdt.keys():
        entries = crdt._entries.get(key, [])
        if entries:
            best = max(entries, key=lambda e: e.tag)
            digest[key] = tuple(best.tag)
    return digest


def encode_message(msg_type: str, payload: dict, reply_port: int = 0) -> bytes:
    """Encode a gossip message to wire format.

    reply_port: The sender's listening port, so the receiver can
    register the sender for future gossip (peer discovery).
    """
    msg = {"type": msg_type, "reply_port": reply_port, **payload}
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode_message(data: bytes) -> Optional[dict]:
    """Decode a gossip message from wire format. Returns None on error.

    Enforces MAX_MESSAGE_BYTES to prevent memory exhaustion from
    oversized payloads before JSON parsing.
    """
    if len(data) > MAX_MESSAGE_BYTES:
        logger.warning(
            "Rejecting oversized gossip message: %d bytes (limit %d)",
            len(data), MAX_MESSAGE_BYTES,
        )
        return None
    try:
        msg = json.loads(data)
        if not isinstance(msg, dict) or "type" not in msg:
            return None
        return msg
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def build_full_state_message(crdt: ORMap, reply_port: int = 0) -> bytes:
    """Build a FULL_STATE message (complete snapshot)."""
    return encode_message(MSG_FULL_STATE, {
        "snapshot": crdt.snapshot(),
    }, reply_port=reply_port)


def build_digest_message(crdt: ORMap, reply_port: int = 0) -> bytes:
    """Build a DIGEST message for pull-based exchange."""
    return encode_message(MSG_DIGEST, {
        "digest": build_digest(crdt),
        "tombstone_count": crdt.tombstone_count,
    }, reply_port=reply_port)


def build_delta_message(crdt: ORMap, missing_keys: List[str], we_need: Optional[List[str]] = None) -> bytes:
    """Build a DELTA message containing only requested keys + all tombstones.

    This is more efficient than sending the full snapshot when only
    a few keys differ between peers.
    """
    delta_entries = {}
    for k in missing_keys:
        entries = crdt._entries.get(k)
        if entries:
            delta_entries[k] = [{"value": e.value, "tag": list(e.tag)} for e in entries]

    payload = {
        "snapshot": {
            "entries": delta_entries,
            "tombstones": [list(t) for t in crdt._tombstones],
        }
    }
    if we_need:
        payload["we_need"] = we_need

    return encode_message(MSG_DELTA, payload)


def build_digest_response(crdt: ORMap, remote_digest: dict) -> bytes:
    """Compare remote digest with local state. Returns:
    - A DELTA message with entries the remote is missing + a list of keys WE need.

    This is the core of push-pull: both sides figure out what the other needs.
    """
    local_digest = build_digest(crdt)

    remote_needs = []
    we_need = []

    # Keys WE are missing
    for key, remote_tag in remote_digest.items():
        if not isinstance(key, str) or not isinstance(remote_tag, list) or len(remote_tag) != 2:
            continue
        local_tag = local_digest.get(key)
        if local_tag is None or tuple(local_tag) < tuple(remote_tag):
            we_need.append(key)

    # Keys remote is missing or has older versions of
    for key, local_tag in local_digest.items():
        remote_tag = remote_digest.get(key)
        if remote_tag is None or not isinstance(remote_tag, list) or len(remote_tag) != 2:
            remote_needs.append(key)
        elif tuple(remote_tag) < tuple(local_tag):
            remote_needs.append(key)

    # Build delta for what remote needs, and include our wishlist
    return build_delta_message(crdt, remote_needs, we_need=we_need)


def compute_jitter(base_ms: float, jitter_ms: float = JITTER_MS) -> float:
    """Add random jitter to prevent thundering herd."""
    return base_ms + random.uniform(-jitter_ms, jitter_ms)


def handle_incoming_message(
    crdt: ORMap,
    data: bytes,
    stats: GossipStats,
    reply_port: int = 0,
) -> Tuple[Optional[bytes], int]:
    """Process an incoming gossip message.

    Returns (response_bytes, sender_listen_port).
    response_bytes: bytes to send back (for DIGEST messages), or None.
    sender_listen_port: the sender's gossip listening port (for peer discovery).

    Message handling:
    - FULL_STATE → merge directly, no response
    - DIGEST → compare and respond with DELTA
    - DELTA → merge directly, no response
    """
    msg = decode_message(data)
    if msg is None:
        logger.debug("Received invalid gossip message, ignoring")
        return None, 0

    # P2-05: Validate reply_port is within valid TCP range
    sender_port = msg.get("reply_port", 0)
    if not isinstance(sender_port, int) or sender_port < 0 or sender_port > 65535:
        sender_port = 0

    msg_type = msg.get("type")

    if msg_type == MSG_FULL_STATE:
        snapshot = msg.get("snapshot")
        if snapshot:
            delta = crdt.merge(snapshot)
            stats.merges += 1
            if delta > 0:
                stats.delta_keys_received += delta
        return None, sender_port

    elif msg_type == MSG_DIGEST:
        remote_digest = msg.get("digest", {})
        if not isinstance(remote_digest, dict):
            remote_digest = {}
        response_delta = build_digest_response(crdt, remote_digest)
        stats.pull_sent += 1
        return response_delta, sender_port

    elif msg_type in (MSG_DELTA, MSG_DIGEST_RESPONSE):
        snapshot = msg.get("snapshot")
        # Validate snapshot is a dict to avoid crash on merge
        if snapshot is not None and isinstance(snapshot, dict):
            delta = crdt.merge(snapshot)
            stats.merges += 1
            if delta > 0:
                stats.delta_keys_received += delta

        # If the sender also told us what they need, fulfill it!
        we_need_remote = msg.get("we_need")
        if isinstance(we_need_remote, list) and we_need_remote:
            # Send another DELTA back to fulfill their request
            response_delta = build_delta_message(crdt, we_need_remote)
            return response_delta, sender_port

        return None, sender_port

    else:
        logger.debug("Unknown gossip message type: %s", msg_type)
        return None, 0
