"""propagul.a2a.room — AgentRoom: Collaboration context management.

An AgentRoom groups agents working on a shared task.
It manages the SharedAgentState lifecycle and provides
peer discovery and membership tracking.

Usage:
    room = AgentRoom("project-alpha")
    room.join("agent-01", capabilities=["inference", "rag"])
    room.join("agent-02", capabilities=["search"])

    state = room.state  # SharedAgentState instance
    state.set("query", "What is CRDT?")

    members = room.members()  # List of AgentInfo
    room.leave("agent-02")
"""

import logging
import time
from typing import Dict, List, Optional

from propagul.a2a.state import SharedAgentState
from propagul.a2a.types import A2AConfig, AgentInfo

logger = logging.getLogger("propagul.a2a.room")


class AgentRoom:
    """Collaboration context for multi-agent workflows.

    Manages state and membership for a group of agents
    working on a shared task or conversation.

    Args:
        room_id: Unique room/channel identifier.
        config: Optional A2AConfig for persistence and tuning.
    """

    def __init__(
        self,
        room_id: str,
        config: Optional[A2AConfig] = None,
    ) -> None:
        self.room_id = room_id
        self._config = config or A2AConfig()
        self._members: Dict[str, AgentInfo] = {}
        self._state: Optional[SharedAgentState] = None
        self._created_at = time.time()

    # ─── Membership ───────────────────────────────────────────────

    def join(
        self,
        agent_id: str,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> AgentInfo:
        """Add an agent to this room.

        Creates the SharedAgentState on first join (lazy init).
        If the agent already exists, updates their info.

        Args:
            agent_id: Unique agent identifier.
            capabilities: Optional list of agent capabilities.
            metadata: Optional key-value metadata.

        Returns:
            AgentInfo for the joined agent.
        """
        info = AgentInfo(
            agent_id=agent_id,
            last_seen=time.time(),
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        self._members[agent_id] = info

        # Lazy-init shared state on first join
        if self._state is None:
            self._state = SharedAgentState(
                room=self.room_id,
                agent_id=agent_id,
                config=self._config,
            )

        # Register presence in CRDT
        self._state.set_meta(f"member:{agent_id}", "joined")
        self._state.set_meta(
            f"member:{agent_id}:caps",
            ",".join(capabilities or []),
        )

        logger.info("Agent %s joined room %s", agent_id, self.room_id)
        return info

    def leave(self, agent_id: str) -> bool:
        """Remove an agent from this room.

        Returns True if the agent was a member.
        Does NOT delete the agent's state from CRDT (preserves history).
        """
        if agent_id not in self._members:
            return False

        del self._members[agent_id]

        if self._state:
            self._state.set_meta(f"member:{agent_id}", "left")

        logger.info("Agent %s left room %s", agent_id, self.room_id)
        return True

    def members(self) -> List[AgentInfo]:
        """Get all current room members."""
        return list(self._members.values())

    def active_members(self) -> List[AgentInfo]:
        """Get members that are still active (within TTL)."""
        return [m for m in self._members.values() if m.is_active]

    def member_count(self) -> int:
        """Number of current members."""
        return len(self._members)

    def has_member(self, agent_id: str) -> bool:
        """Check if an agent is a member."""
        return agent_id in self._members

    # ─── State Access ─────────────────────────────────────────────

    @property
    def state(self) -> Optional[SharedAgentState]:
        """Get the room's shared state.

        Returns None if no agents have joined yet.
        """
        return self._state

    # ─── Summary ──────────────────────────────────────────────────

    def summary(self) -> dict:
        """Room summary for dashboard/API."""
        return {
            "room_id": self.room_id,
            "member_count": len(self._members),
            "members": [m.agent_id for m in self._members.values()],
            "created_at": self._created_at,
            "state_keys": self._state.key_count if self._state else 0,
            "has_state": self._state is not None,
        }

    def __repr__(self) -> str:
        return (
            f"AgentRoom(id={self.room_id!r}, members={len(self._members)}, "
            f"keys={self._state.key_count if self._state else 0})"
        )

    # ─── Persistence ──────────────────────────────────────────────

    def flush(self) -> bool:
        """Flush state to disk (for graceful shutdown)."""
        if self._state:
            return self._state.flush_to_disk()
        return False
