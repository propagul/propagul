"""propagul.a2a.types — Type definitions for A2A State Layer.

Dataclasses and type aliases used across the A2A package.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class A2AConfig:
    """Configuration for A2A State Layer.

    Attributes:
        persist_path: Optional disk persistence path for room state.
        debounce_seconds: Write coalescing interval (default 2s).
        agent_ttl_seconds: Agents not seen for this long are considered gone (default 5m).
        max_keys_per_room: Maximum keys per room to prevent unbounded growth.
    """
    persist_path: Optional[str] = None
    debounce_seconds: float = 2.0
    agent_ttl_seconds: float = 300.0
    max_keys_per_room: int = 10_000


@dataclass
class AgentInfo:
    """Metadata about a connected agent.

    Attributes:
        agent_id: Unique agent identifier.
        last_seen: Timestamp of last activity (epoch seconds).
        capabilities: Optional list of agent capabilities/roles.
        metadata: Arbitrary key-value metadata.
    """
    agent_id: str
    last_seen: float = field(default_factory=time.time)
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Agent seen within the last 5 minutes."""
        return (time.time() - self.last_seen) < 300

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "last_seen": self.last_seen,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }


@dataclass
class SyncResult:
    """Result of a CRDT merge operation.

    Attributes:
        delta: Number of new/updated entries from the merge.
        local_keys: Total keys in local state after merge.
        remote_keys: Number of keys in the remote snapshot.
        conflicts_resolved: Number of LWW conflicts resolved.
    """
    delta: int = 0
    local_keys: int = 0
    remote_keys: int = 0
    conflicts_resolved: int = 0

    @property
    def had_changes(self) -> bool:
        return self.delta > 0

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "local_keys": self.local_keys,
            "remote_keys": self.remote_keys,
            "conflicts_resolved": self.conflicts_resolved,
            "had_changes": self.had_changes,
        }
