"""propagul.mesh.router — Intelligent fleet-aware request routing.

Implements: Least-Connections + Model-Affinity + Automatic Failover.

Routing algorithm:
    1. Request arrives for model "llama3.1:8b"
    2. Check fleet state: which online nodes have this model?
    3. If multiple candidates: pick the one with fewest active connections
    4. If no candidate has the model: pick node with most free VRAM
    5. If selected node fails (timeout/5xx): retry with next candidate
    6. If no fleet state available: fall back to local backend (backward compat)

Data flow:
    Dashboard heartbeat response → Agent parses fleet routing table →
    Agent updates FleetState → Proxy reads FleetState for each request →
    Proxy routes directly to remote backend in LAN (prompt never touches cloud)

Zero external dependencies. Pure stdlib.
Thread-safe: FleetState is read by asyncio proxy, written by heartbeat thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("propagul.mesh.router")


@dataclass(frozen=True)
class BackendTarget:
    """A routable backend endpoint in the fleet.

    Represents one backend on one node. A node can have multiple backends
    (e.g. Ollama + vLLM), each with its own URL and model list.
    """
    node_id: str
    backend_name: str     # "ollama", "vllm", "tgi", "lm_studio", "llama_cpp"
    backend_url: str      # "http://192.168.1.10:11434"
    models: tuple         # Tuple of model name strings (frozen for hashing)
    free_vram_mb: float   # Available VRAM on this node
    backend_auth: str = ""  # Optional auth header for this backend

    def has_model(self, model: str) -> bool:
        """Check if this backend has the requested model loaded."""
        return model in self.models


class ActiveConnectionTracker:
    """Thread-safe per-node active connection counter.

    Used for least-connections routing. Incremented when a request starts,
    decremented when it completes (success or failure).

    Thread-safety: Uses threading.Lock (not asyncio.Lock) because the proxy
    dispatches HTTP requests in executor threads via run_in_executor.
    """

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, node_id: str) -> None:
        """Increment active connections for a node."""
        with self._lock:
            self._counts[node_id] = self._counts.get(node_id, 0) + 1

    def release(self, node_id: str) -> None:
        """Decrement active connections for a node."""
        with self._lock:
            current = self._counts.get(node_id, 0)
            if current > 0:
                self._counts[node_id] = current - 1

    def count(self, node_id: str) -> int:
        """Get current active connection count (lock-free read, approximate)."""
        return self._counts.get(node_id, 0)

    def snapshot(self) -> dict[str, int]:
        """Get a snapshot of all counts (for logging/diagnostics)."""
        with self._lock:
            return dict(self._counts)


@dataclass(frozen=True)
class RoutingEvent:
    """A single routing decision record.

    Captured by the proxy for each inference request, drained by the agent
    and sent to the dashboard in the heartbeat telemetry.
    """
    timestamp: float
    model: str
    target_node: str
    target_backend: str
    reason: str  # "model_affinity", "least_connections", "vram_fallback", "local_fallback", "fleet_error"


class RoutingEventLog:
    """Thread-safe ring buffer for routing events.

    Written by the asyncio proxy (from the event loop thread).
    Read/drained by the heartbeat thread (different thread).
    """

    def __init__(self, maxlen: int = 50):
        self._events: List[RoutingEvent] = []
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def record(self, event: RoutingEvent) -> None:
        """Append a routing event. Oldest events are discarded when full."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._maxlen:
                self._events = self._events[-self._maxlen:]

    def drain(self) -> List[dict]:
        """Get all events as dicts and clear the buffer.

        Called by the heartbeat thread to include events in telemetry.
        Returns list of plain dicts (JSON-serializable).
        """
        with self._lock:
            events = [
                {
                    "ts": e.timestamp,
                    "model": e.model,
                    "target_node": e.target_node,
                    "target_backend": e.target_backend,
                    "reason": e.reason,
                }
                for e in self._events
            ]
            self._events = []
            return events


class FleetState:
    """Thread-safe snapshot of fleet node states for routing.

    Updated by the agent's heartbeat loop (write thread).
    Read by the proxy's request handler (asyncio event loop + executor threads).

    The fleet state is a simple list of BackendTargets. It is replaced
    atomically (reference swap) on each heartbeat, so readers never see
    a partially-updated state.
    """

    def __init__(self):
        self._targets: tuple[BackendTarget, ...] = ()
        self._updated_at: float = 0.0
        self._lock = threading.Lock()

    def update(self, targets: List[BackendTarget]) -> None:
        """Replace the fleet state atomically.

        Called by the agent after parsing the heartbeat response.
        """
        with self._lock:
            self._targets = tuple(targets)
            self._updated_at = time.time()
        logger.debug(
            "Fleet state updated: %d targets across %d nodes",
            len(targets),
            len(set(t.node_id for t in targets)),
        )

    @property
    def targets(self) -> tuple[BackendTarget, ...]:
        """Get current fleet targets (lock-free read of immutable tuple)."""
        return self._targets

    @property
    def age_seconds(self) -> float:
        """How old is the fleet state (seconds since last update)."""
        if self._updated_at == 0.0:
            return float("inf")
        return time.time() - self._updated_at

    @property
    def is_stale(self) -> bool:
        """Fleet state is stale if older than 2 heartbeat intervals (60s)."""
        return self.age_seconds > 60.0

    def nodes_with_model(self, model: str) -> List[BackendTarget]:
        """Get all online targets that have the requested model loaded."""
        return [t for t in self._targets if t.has_model(model)]

    def nodes_by_free_vram(self) -> List[BackendTarget]:
        """Get all targets sorted by free VRAM (most free first)."""
        return sorted(self._targets, key=lambda t: t.free_vram_mb, reverse=True)


# Maximum staleness before we ignore fleet state entirely
_MAX_FLEET_AGE_SECONDS = 120.0


class RequestRouter:
    """Selects the best backend for each inference request.

    Routing priority:
        1. Model-Affinity: prefer nodes that already have the model loaded
        2. Least-Connections: among affinity candidates, pick fewest active requests
        3. VRAM-Fallback: if no node has the model, pick node with most free VRAM
        4. Local-Fallback: if fleet state is empty/stale, use local backend

    Automatic failover: if the selected backend fails, the caller retries with
    the next candidate from the ordered list.
    """

    def __init__(
        self,
        fleet_state: FleetState,
        local_node_id: str = "",
        tracker: Optional[ActiveConnectionTracker] = None,
    ):
        self._fleet = fleet_state
        self._local_node_id = local_node_id
        self._tracker = tracker or ActiveConnectionTracker()

    @property
    def tracker(self) -> ActiveConnectionTracker:
        return self._tracker

    def select(self, model: str) -> Optional[BackendTarget]:
        """Select the best backend for the given model.

        Returns None if no candidate is available (caller should use
        local backend as fallback).
        """
        if self._fleet.is_stale or not self._fleet.targets:
            return None  # Fall back to local

        candidates = self._fleet.nodes_with_model(model)

        if not candidates:
            # No node has this model loaded → route to most free VRAM
            candidates = self._fleet.nodes_by_free_vram()
            if candidates:
                selected = candidates[0]
                logger.info(
                    "Model '%s' not loaded anywhere. "
                    "Routing to node '%s' (%.0f MB free VRAM) — "
                    "it will need to load the model.",
                    model, selected.node_id, selected.free_vram_mb,
                )
                return selected
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates with the model → least-connections tiebreak
        # Prefer local node if connection count is equal (avoid network hop)
        def sort_key(t: BackendTarget) -> tuple:
            active = self._tracker.count(t.node_id)
            is_remote = 0 if t.node_id == self._local_node_id else 1
            return (active, is_remote)

        candidates.sort(key=sort_key)
        selected = candidates[0]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Routing '%s' → node '%s' (%d active, %d candidates)",
                model, selected.node_id,
                self._tracker.count(selected.node_id),
                len(candidates),
            )

        return selected

    def select_ordered(self, model: str) -> List[BackendTarget]:
        """Get all candidates ordered by preference (for failover).

        First element is the preferred backend. If it fails, caller
        should try the next one.
        """
        if self._fleet.is_stale or not self._fleet.targets:
            return []

        candidates = self._fleet.nodes_with_model(model)
        if not candidates:
            candidates = self._fleet.nodes_by_free_vram()

        def sort_key(t: BackendTarget) -> tuple:
            active = self._tracker.count(t.node_id)
            is_remote = 0 if t.node_id == self._local_node_id else 1
            return (active, is_remote)

        candidates.sort(key=sort_key)
        return candidates


def parse_fleet_routing_table(data: dict) -> List[BackendTarget]:
    """Parse the fleet routing table from a heartbeat response.

    Expected format:
        {
            "fleet_routing": [
                {
                    "node_id": "gpu-1",
                    "backend_name": "ollama",
                    "backend_url": "http://192.168.1.10:11434",
                    "models": ["llama3.1:8b", "mistral:7b"],
                    "free_vram_mb": 8192.0,
                    "backend_auth": ""
                },
                ...
            ]
        }
    """
    targets = []
    for entry in data.get("fleet_routing", []):
        try:
            targets.append(BackendTarget(
                node_id=entry["node_id"],
                backend_name=entry.get("backend_name", "unknown"),
                backend_url=entry["backend_url"],
                models=tuple(entry.get("models", [])),
                free_vram_mb=float(entry.get("free_vram_mb", 0.0)),
                backend_auth=entry.get("backend_auth", ""),
            ))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping invalid fleet routing entry: %s", e)
            continue

    return targets
