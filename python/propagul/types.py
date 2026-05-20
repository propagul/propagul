"""Type definitions for propagul SDK."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any


@dataclass
class PeerAddress:
    """Network address of a gossip peer."""
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def __hash__(self) -> int:
        return hash((self.host, self.port))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PeerAddress):
            return NotImplemented
        return self.host == other.host and self.port == other.port


@dataclass
class StoreStats:
    """Telemetry snapshot from the AgentStateStore."""
    entropy: float = 0.0
    current_k: int = 1
    sleep_ratio: float = 0.0
    loss_rate: float = 0.0
    convergence_ms: float = 0.0
    is_partitioned: bool = False
    shock_events: int = 0
    key_count: int = 0
    tombstone_count: int = 0
    peer_count: int = 0
    gossip_rounds: int = 0


@dataclass
class QoSTier:
    """Quality-of-Service tier for gossip scheduling."""
    name: str
    prefix: str
    gossip_interval_ms: float = 500.0
    priority: int = 1  # 1=low, 3=high


# Default QoS tiers
QOS_FAST = QoSTier(name="fast", prefix="__fast__/", gossip_interval_ms=100.0, priority=3)
QOS_NORMAL = QoSTier(name="normal", prefix="", gossip_interval_ms=500.0, priority=2)
QOS_BULK = QoSTier(name="bulk", prefix="__bulk__/", gossip_interval_ms=2000.0, priority=1)

DEFAULT_QOS_TIERS = [QOS_FAST, QOS_NORMAL, QOS_BULK]

# Callback types
ChangeCallback = Callable[[str, Optional[str]], None]  # (key, new_value)
