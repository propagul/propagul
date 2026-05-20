"""propagul.a2a.state — SharedAgentState: CRDT-backed shared state.

Core abstraction for multi-agent state sharing. Wraps an ORMap CRDT
with namespace isolation, agent presence tracking, and optional persistence.

Namespaces:
    agent:{id}:       — Per-agent private state (only that agent writes)
    shared:           — Shared state (any agent can write)
    meta:             — Room metadata (system-managed)

Conflict Resolution:
    LWW (Last-Writer-Wins) via CRDT timestamps. Concurrent writes to
    the same key resolve deterministically based on node_id ordering.

Thread Safety:
    Single-threaded (async event loop). NOT safe for multi-thread access.
    Use asyncio.Lock if needed in concurrent scenarios.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from propagul.crdt import ORMap
from propagul.a2a.types import A2AConfig, AgentInfo, SyncResult

logger = logging.getLogger("propagul.a2a.state")

# Namespace prefixes
_NS_AGENT = "agent:"
_NS_SHARED = "shared:"
_NS_META = "meta:"


class SharedAgentState:
    """CRDT-backed shared state for multi-agent collaboration.

    Each agent gets an isolated namespace for private state,
    plus access to a shared namespace for coordination.

    Args:
        room: Room/channel name for isolation.
        agent_id: This agent's unique identifier.
        node_id: Numeric CRDT node ID (hashed from agent_id if not provided).
        config: Optional A2AConfig for persistence and tuning.
    """

    def __init__(
        self,
        room: str,
        agent_id: str,
        node_id: Optional[int] = None,
        config: Optional[A2AConfig] = None,
    ) -> None:
        self.room = room
        self.agent_id = agent_id
        self._config = config or A2AConfig()

        # Derive numeric node_id from agent_id if not provided
        if node_id is None:
            node_id = hash(agent_id) % (2**31)  # Positive 31-bit int
        self._node_id = node_id

        self._crdt = ORMap(node_id)
        self._agents: Dict[str, AgentInfo] = {}

        # Persistence (reuses config_sync patterns)
        self._persist_path: Optional[str] = None
        if self._config.persist_path:
            self._persist_path = os.path.join(
                self._config.persist_path,
                f"a2a_room_{room}.json",
            )
            self._load_from_disk()

        self._last_save_time: float = 0.0
        self._dirty = False

        # Register self as active agent
        self._touch_presence()

    # ─── Public API: Shared State ──────────────────────────────────

    def set(self, key: str, value: str) -> None:
        """Set a shared key-value pair.

        Writes to the 'shared:' namespace, visible to all agents.

        Args:
            key: State key (e.g., "task_status", "progress").
            value: State value as string. Use JSON for complex values.

        Raises:
            ValueError: If max_keys_per_room exceeded.
        """
        if len(self._crdt) >= self._config.max_keys_per_room:
            raise ValueError(
                f"Max keys ({self._config.max_keys_per_room}) reached for room '{self.room}'"
            )
        full_key = f"{_NS_SHARED}{key}"
        self._crdt.set(full_key, value)
        self._touch_presence()
        self._auto_save()

    def get(self, key: str) -> Optional[str]:
        """Get a shared value by key.

        Returns None if the key doesn't exist or has been deleted.
        """
        full_key = f"{_NS_SHARED}{key}"
        return self._crdt.get(full_key)

    def delete(self, key: str) -> None:
        """Delete a shared key (tombstoned in CRDT)."""
        full_key = f"{_NS_SHARED}{key}"
        self._crdt.delete(full_key)
        self._auto_save()

    def get_all_shared(self) -> Dict[str, str]:
        """Get all shared state as a dict."""
        result = {}
        for key in self._crdt.keys():
            if key.startswith(_NS_SHARED):
                short_key = key[len(_NS_SHARED):]
                val = self._crdt.get(key)
                if val is not None:
                    result[short_key] = val
        return result

    # ─── Public API: Agent Private State ──────────────────────────

    def set_private(self, key: str, value: str) -> None:
        """Set a private key-value pair for this agent.

        Only visible to this agent (namespace: agent:{agent_id}:).
        Other agents can read it via get_agent_state().
        """
        full_key = f"{_NS_AGENT}{self.agent_id}:{key}"
        self._crdt.set(full_key, value)
        self._touch_presence()
        self._auto_save()

    def get_private(self, key: str) -> Optional[str]:
        """Get a private value for this agent."""
        full_key = f"{_NS_AGENT}{self.agent_id}:{key}"
        return self._crdt.get(full_key)

    def get_agent_state(self, agent_id: str) -> Dict[str, str]:
        """Read another agent's state (read-only view)."""
        prefix = f"{_NS_AGENT}{agent_id}:"
        result = {}
        for key in self._crdt.keys():
            if key.startswith(prefix):
                short_key = key[len(prefix):]
                val = self._crdt.get(key)
                if val is not None:
                    result[short_key] = val
        return result

    # ─── Public API: Metadata ─────────────────────────────────────

    def set_meta(self, key: str, value: str) -> None:
        """Set room metadata."""
        full_key = f"{_NS_META}{key}"
        self._crdt.set(full_key, value)
        self._auto_save()

    def get_meta(self, key: str) -> Optional[str]:
        """Get room metadata."""
        full_key = f"{_NS_META}{key}"
        return self._crdt.get(full_key)

    # ─── CRDT Sync ────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Get CRDT snapshot for sync over the wire."""
        return self._crdt.snapshot()

    def merge(self, remote_snapshot: dict) -> SyncResult:
        """Merge a remote state snapshot.

        CRDT guarantees: commutative, associative, idempotent.
        After merge, both sides converge to the same state.

        Returns:
            SyncResult with delta and key counts.
        """
        remote_keys = len(remote_snapshot.get("entries", {}))
        delta = self._crdt.merge(remote_snapshot)
        result = SyncResult(
            delta=delta,
            local_keys=len(self._crdt),
            remote_keys=remote_keys,
        )
        if delta > 0:
            self._auto_save()
        return result

    # ─── Agent Presence ───────────────────────────────────────────

    def _touch_presence(self) -> None:
        """Update this agent's presence timestamp."""
        ts = str(time.time())
        self._crdt.set(f"{_NS_META}agent_seen:{self.agent_id}", ts)

        if self.agent_id not in self._agents:
            self._agents[self.agent_id] = AgentInfo(agent_id=self.agent_id)
        self._agents[self.agent_id].last_seen = time.time()

    def get_active_agents(self) -> List[AgentInfo]:
        """Get list of agents that have been active recently.

        Uses the agent_ttl_seconds from config (default 5 minutes).
        """
        now = time.time()
        ttl = self._config.agent_ttl_seconds
        active = []

        for key in self._crdt.keys():
            prefix = f"{_NS_META}agent_seen:"
            if key.startswith(prefix):
                agent_id = key[len(prefix):]
                ts_str = self._crdt.get(key)
                if ts_str is None:
                    continue
                try:
                    last_seen = float(ts_str)
                except (ValueError, TypeError):
                    continue

                if (now - last_seen) < ttl:
                    info = self._agents.get(agent_id, AgentInfo(agent_id=agent_id))
                    info.last_seen = last_seen
                    active.append(info)

        return active

    # ─── Summary ──────────────────────────────────────────────────

    @property
    def key_count(self) -> int:
        """Total number of active keys."""
        return len(self._crdt)

    def summary(self) -> dict:
        """Human-readable state summary."""
        shared = self.get_all_shared()
        agents = self.get_active_agents()
        return {
            "room": self.room,
            "agent_id": self.agent_id,
            "total_keys": len(self._crdt),
            "shared_keys": len(shared),
            "active_agents": len(agents),
            "agent_ids": [a.agent_id for a in agents],
        }

    def __repr__(self) -> str:
        return (
            f"SharedAgentState(room={self.room!r}, agent={self.agent_id!r}, "
            f"keys={len(self._crdt)})"
        )

    # ─── Persistence ──────────────────────────────────────────────

    def _auto_save(self) -> None:
        """Debounced save to disk."""
        if not self._persist_path:
            return

        self._dirty = True
        now = time.monotonic()
        if (now - self._last_save_time) >= self._config.debounce_seconds:
            self.save_to_disk()

    def save_to_disk(self) -> bool:
        """Save CRDT snapshot to disk (atomic write)."""
        if not self._persist_path:
            return False

        data = {
            "version": 1,
            "room": self.room,
            "snapshot": self._crdt.snapshot(),
        }

        tmp_path = self._persist_path + ".tmp"
        try:
            parent = os.path.dirname(self._persist_path) or "."
            os.makedirs(parent, exist_ok=True)

            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._persist_path)
            self._last_save_time = time.monotonic()
            self._dirty = False
            logger.info("A2A state saved: room=%s, keys=%d", self.room, len(self._crdt))
            return True
        except Exception as e:
            logger.error("Failed to save A2A state for room=%s: %s", self.room, e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False

    def _load_from_disk(self) -> int:
        """Load state from disk and merge."""
        if not self._persist_path or not os.path.isfile(self._persist_path):
            return 0

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            if data.get("version") != 1:
                logger.warning("Unknown A2A state version: %s", data.get("version"))
                return 0

            snapshot = data.get("snapshot")
            if not isinstance(snapshot, dict):
                return 0

            delta = self._crdt.merge(snapshot)
            logger.info(
                "A2A state loaded: room=%s, keys=%d, delta=%d",
                self.room, len(self._crdt), delta,
            )
            return delta

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Corrupt A2A state file for room=%s: %s", self.room, e)
            return 0

    def flush_to_disk(self) -> bool:
        """Force-save state to disk (for graceful shutdown).

        Always writes if persist_path is configured, regardless of dirty flag.
        Use this on shutdown to guarantee no data loss.
        """
        if self._persist_path:
            return self.save_to_disk()
        return False

    @property
    def is_dirty(self) -> bool:
        """True if unsaved mutations exist."""
        return self._dirty
