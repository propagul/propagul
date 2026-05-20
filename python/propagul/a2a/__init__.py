"""propagul.a2a — Shared Agent State Layer.

CRDT-backed shared state for multi-agent workflows.
Allows AI agents to share key-value state across the Propagul network
without central coordination.

Architecture:
    SharedAgentState wraps an ORMap CRDT with namespace isolation.
    AgentRoom manages a collaboration context (state + peer discovery).

Example:
    from propagul.a2a import SharedAgentState

    state = SharedAgentState(room="project-alpha", agent_id="agent-01")
    state.set("task_status", "processing")
    state.set("progress", "0.75")
    snapshot = state.snapshot()
    state.merge(remote_snapshot)
"""

from propagul.a2a.state import SharedAgentState
from propagul.a2a.room import AgentRoom
from propagul.a2a.types import A2AConfig, AgentInfo, SyncResult

__all__ = [
    "SharedAgentState",
    "AgentRoom",
    "A2AConfig",
    "AgentInfo",
    "SyncResult",
]
