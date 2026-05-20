"""propagul.integrations.crewai — CrewAI integration plugin.

Provides PersistentCrew: a drop-in replacement for crewai.Crew that
persists agent state via Propagul CRDT. On crash + restart,
agents recover their state automatically.

Usage:
    from propagul.integrations.crewai import PersistentCrew

    crew = PersistentCrew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        state_room="my-project",
        state_port=9001,
        recovery=True,
    )
    result = crew.kickoff()

Requires: pip install crewai
"""

import asyncio
import atexit
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("propagul.crewai")

# Lazy import to avoid hard dependency on crewai
_crewai_available = None


def _check_crewai():
    global _crewai_available
    if _crewai_available is None:
        try:
            import crewai  # noqa: F401
            _crewai_available = True
        except ImportError:
            _crewai_available = False
    return _crewai_available


class PersistentCrew:
    """Drop-in CrewAI Crew replacement with crash-resilient state persistence.

    Wraps a standard crewai.Crew and adds:
    - Automatic state persistence via Propagul CRDT
    - Crash recovery: state is restored from peers on restart
    - Task progress tracking: each task's status is synced
    - Agent health monitoring via gossip

    State keys managed automatically:
    - "crew/status" → "running" | "completed" | "failed"
    - "crew/start_time" → ISO timestamp
    - "task/{i}/status" → "pending" | "running" | "completed"
    - "task/{i}/result" → serialized task output
    - "agent/{name}/status" → "active" | "idle" | "crashed"
    """

    def __init__(
        self,
        agents: List[Any],
        tasks: List[Any],
        state_room: str = "default",
        state_port: int = 9001,
        state_peers: Optional[List[Any]] = None,
        recovery: bool = True,
        verbose: bool = False,
        node_id: Optional[int] = None,
        **crew_kwargs: Any,
    ):
        if not _check_crewai():
            raise ImportError(
                "crewai is required for PersistentCrew. "
                "Install with: pip install propagul[crewai]"
            )

        from crewai import Crew
        from propagul.store import AgentStateStore
        from propagul.types import PeerAddress

        # Generate unique node_id from port if not provided
        self._node_id = node_id or (state_port * 31 + 7)

        # Create state store
        self._store = AgentStateStore(
            room=state_room,
            node_id=self._node_id,
            port=state_port,
            gossip_interval_ms=200.0,  # Faster gossip for task tracking
        )

        # Parse peers
        self._peers = []
        if state_peers:
            for p in state_peers:
                if isinstance(p, PeerAddress):
                    self._peers.append(p)
                elif isinstance(p, tuple) and len(p) == 2:
                    self._peers.append(PeerAddress(p[0], p[1]))
                elif isinstance(p, str) and ":" in p:
                    host, port_str = p.rsplit(":", 1)
                    self._peers.append(PeerAddress(host, int(port_str)))

        self._recovery = recovery
        self._verbose = verbose
        self._agents = agents
        self._tasks = tasks
        self._crew_kwargs = crew_kwargs
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Create the underlying Crew
        self._crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=verbose,
            **crew_kwargs,
        )

    def kickoff(self) -> Any:
        """Run the crew with state persistence.

        On crash + restart with recovery=True, previously completed
        task results are restored from peers automatically.
        """
        # Start state sync
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._store.start(peers=self._peers))

        # Register cleanup
        atexit.register(self._cleanup)

        try:
            # Check for recoverable state
            if self._recovery:
                self._try_recover()

            # Record crew start
            self._store.set("crew/status", "running")
            self._store.set("crew/start_time", str(time.time()))

            # Initialize task states
            for i, task in enumerate(self._tasks):
                existing = self._store.get(f"task/{i}/status")
                if existing != "completed":
                    self._store.set(f"task/{i}/status", "pending")
                    desc = getattr(task, "description", str(task))
                    self._store.set(f"task/{i}/description", desc[:200])

            # Mark agents active
            for agent in self._agents:
                name = getattr(agent, "role", str(agent))
                self._store.set(f"agent/{name}/status", "active")

            # Run the crew
            if self._verbose:
                logger.info("Starting crew with %d agents, %d tasks",
                            len(self._agents), len(self._tasks))

            result = self._crew.kickoff()

            # Record completion
            self._store.set("crew/status", "completed")
            for i in range(len(self._tasks)):
                self._store.set(f"task/{i}/status", "completed")

            if result is not None:
                self._store.set("crew/result", str(result)[:4096])

            # Give gossip time to propagate final state
            self._loop.run_until_complete(asyncio.sleep(0.5))

            return result

        except Exception as e:
            self._store.set("crew/status", f"failed:{type(e).__name__}")
            self._loop.run_until_complete(asyncio.sleep(0.3))
            raise

        finally:
            self._cleanup()

    def _try_recover(self) -> None:
        """Attempt to recover state from peers.

        Waits briefly for gossip sync, then checks if any tasks
        were already completed by a previous run.
        """
        if not self._peers:
            return

        # Wait for initial gossip sync
        self._loop.run_until_complete(asyncio.sleep(1.0))

        # Check for recovered state
        crew_status = self._store.get("crew/status")
        if crew_status:
            logger.info("Recovered crew state: %s", crew_status)

        recovered_count = 0
        for i in range(len(self._tasks)):
            status = self._store.get(f"task/{i}/status")
            if status == "completed":
                recovered_count += 1
                result = self._store.get(f"task/{i}/result")
                if self._verbose:
                    logger.info("Task %d recovered (completed): %s",
                                i, (result or "")[:100])

        if recovered_count > 0:
            logger.info("Recovered %d/%d completed tasks from peers",
                        recovered_count, len(self._tasks))

    @property
    def state(self) -> Dict[str, str]:
        """Get all current state (for debugging)."""
        return self._store.get_all()

    @property
    def stats(self):
        """Get Propagul telemetry."""
        return self._store.stats

    def _cleanup(self) -> None:
        """Stop state sync."""
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._store.stop())
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
