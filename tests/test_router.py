"""Tests for propagul.mesh.router — Fleet-aware routing logic.

Tests cover:
- BackendTarget model matching
- ActiveConnectionTracker thread-safety
- FleetState atomic updates and staleness
- RequestRouter: model-affinity, least-connections, VRAM-fallback, local-preference
- parse_fleet_routing_table with valid and invalid data
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from propagul.mesh.router import (
    BackendTarget,
    ActiveConnectionTracker,
    FleetState,
    RequestRouter,
    parse_fleet_routing_table,
)


# ═══════════════════════════════════════════════════════════════
# 1. BackendTarget
# ═══════════════════════════════════════════════════════════════

def test_backend_target_has_model():
    t = BackendTarget(
        node_id="gpu-1", backend_name="ollama",
        backend_url="http://192.168.1.10:11434",
        models=("llama3.1:8b", "mistral:7b"),
        free_vram_mb=8000.0,
    )
    assert t.has_model("llama3.1:8b")
    assert t.has_model("mistral:7b")
    assert not t.has_model("gpt-4")
    assert not t.has_model("")


def test_backend_target_frozen():
    """BackendTarget is frozen (immutable, hashable)."""
    t = BackendTarget(
        node_id="gpu-1", backend_name="ollama",
        backend_url="http://localhost:11434",
        models=("test:latest",), free_vram_mb=1000.0,
    )
    # Should be hashable (usable in sets)
    s = {t}
    assert t in s


# ═══════════════════════════════════════════════════════════════
# 2. ActiveConnectionTracker
# ═══════════════════════════════════════════════════════════════

def test_tracker_basic():
    tracker = ActiveConnectionTracker()
    assert tracker.count("gpu-1") == 0

    tracker.acquire("gpu-1")
    assert tracker.count("gpu-1") == 1

    tracker.acquire("gpu-1")
    assert tracker.count("gpu-1") == 2

    tracker.release("gpu-1")
    assert tracker.count("gpu-1") == 1

    tracker.release("gpu-1")
    assert tracker.count("gpu-1") == 0


def test_tracker_no_underflow():
    """Release below 0 should not go negative."""
    tracker = ActiveConnectionTracker()
    tracker.release("gpu-1")
    assert tracker.count("gpu-1") == 0


def test_tracker_snapshot():
    tracker = ActiveConnectionTracker()
    tracker.acquire("gpu-1")
    tracker.acquire("gpu-2")
    tracker.acquire("gpu-2")

    snap = tracker.snapshot()
    assert snap == {"gpu-1": 1, "gpu-2": 2}


def test_tracker_thread_safety():
    """Concurrent acquire/release should not corrupt counts."""
    tracker = ActiveConnectionTracker()
    n = 1000

    def worker_acquire():
        for _ in range(n):
            tracker.acquire("test")

    def worker_release():
        for _ in range(n):
            tracker.release("test")

    t1 = threading.Thread(target=worker_acquire)
    t2 = threading.Thread(target=worker_release)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Net effect should be 0 (n acquires, n releases)
    assert tracker.count("test") == 0


# ═══════════════════════════════════════════════════════════════
# 3. FleetState
# ═══════════════════════════════════════════════════════════════

def _make_target(node_id, models, free_vram=4000.0, backend="ollama"):
    return BackendTarget(
        node_id=node_id, backend_name=backend,
        backend_url=f"http://192.168.1.{hash(node_id) % 254 + 1}:11434",
        models=tuple(models), free_vram_mb=free_vram,
    )


def test_fleet_state_empty():
    fs = FleetState()
    assert fs.targets == ()
    assert fs.is_stale
    assert fs.nodes_with_model("any") == []


def test_fleet_state_update():
    fs = FleetState()
    targets = [_make_target("gpu-1", ["llama3.1:8b"])]
    fs.update(targets)

    assert len(fs.targets) == 1
    assert not fs.is_stale
    assert fs.age_seconds < 1.0


def test_fleet_state_nodes_with_model():
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["llama3.1:8b", "mistral:7b"]),
        _make_target("gpu-2", ["llama3.1:8b"]),
        _make_target("gpu-3", ["gpt-j:6b"]),
    ])

    matches = fs.nodes_with_model("llama3.1:8b")
    assert len(matches) == 2
    assert {m.node_id for m in matches} == {"gpu-1", "gpu-2"}


def test_fleet_state_nodes_by_free_vram():
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["test"], free_vram=2000),
        _make_target("gpu-2", ["test"], free_vram=8000),
        _make_target("gpu-3", ["test"], free_vram=4000),
    ])

    ordered = fs.nodes_by_free_vram()
    assert [t.node_id for t in ordered] == ["gpu-2", "gpu-3", "gpu-1"]


# ═══════════════════════════════════════════════════════════════
# 4. RequestRouter
# ═══════════════════════════════════════════════════════════════

def test_router_model_affinity():
    """Should prefer the node that has the model loaded."""
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["llama3.1:8b"]),
        _make_target("gpu-2", ["mistral:7b"], free_vram=16000),
    ])

    router = RequestRouter(fleet_state=fs, local_node_id="gpu-1")
    selected = router.select("llama3.1:8b")

    assert selected is not None
    assert selected.node_id == "gpu-1"


def test_router_least_connections():
    """Among candidates with the model, pick fewest connections."""
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["llama3.1:8b"]),
        _make_target("gpu-2", ["llama3.1:8b"]),
    ])

    tracker = ActiveConnectionTracker()
    tracker.acquire("gpu-1")
    tracker.acquire("gpu-1")
    tracker.acquire("gpu-1")

    router = RequestRouter(fleet_state=fs, tracker=tracker)
    selected = router.select("llama3.1:8b")

    assert selected is not None
    assert selected.node_id == "gpu-2"  # gpu-1 has 3 active


def test_router_vram_fallback():
    """If no node has the model, route to most free VRAM."""
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["mistral:7b"], free_vram=2000),
        _make_target("gpu-2", ["gpt-j:6b"], free_vram=12000),
    ])

    router = RequestRouter(fleet_state=fs)
    selected = router.select("llama3.1:70b")  # Nobody has it

    assert selected is not None
    assert selected.node_id == "gpu-2"  # Most free VRAM


def test_router_local_preference():
    """If connection count is equal, prefer local node."""
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["llama3.1:8b"]),
        _make_target("gpu-2", ["llama3.1:8b"]),
    ])

    # Both have 0 connections, but gpu-1 is local
    router = RequestRouter(fleet_state=fs, local_node_id="gpu-1")
    selected = router.select("llama3.1:8b")

    assert selected is not None
    assert selected.node_id == "gpu-1"


def test_router_stale_fleet():
    """Returns None when fleet state is stale (fall back to local)."""
    fs = FleetState()
    # Don't update — remains stale
    router = RequestRouter(fleet_state=fs)
    assert router.select("any") is None


def test_router_select_ordered():
    """select_ordered returns all candidates for failover."""
    fs = FleetState()
    fs.update([
        _make_target("gpu-1", ["llama3.1:8b"]),
        _make_target("gpu-2", ["llama3.1:8b"]),
        _make_target("gpu-3", ["llama3.1:8b"]),
    ])

    tracker = ActiveConnectionTracker()
    tracker.acquire("gpu-1")

    router = RequestRouter(fleet_state=fs, tracker=tracker)
    ordered = router.select_ordered("llama3.1:8b")

    assert len(ordered) == 3
    # gpu-1 should be last (1 active connection)
    assert ordered[-1].node_id == "gpu-1"


# ═══════════════════════════════════════════════════════════════
# 5. parse_fleet_routing_table
# ═══════════════════════════════════════════════════════════════

def test_parse_valid():
    data = {
        "fleet_routing": [
            {
                "node_id": "gpu-1",
                "backend_name": "ollama",
                "backend_url": "http://192.168.1.10:11434",
                "models": ["llama3.1:8b", "mistral:7b"],
                "free_vram_mb": 8192.5,
            },
            {
                "node_id": "gpu-2",
                "backend_name": "vllm",
                "backend_url": "http://192.168.1.20:8000",
                "models": ["llama3.1:70b"],
                "free_vram_mb": 40000.0,
                "backend_auth": "Bearer sk-test",
            },
        ]
    }

    targets = parse_fleet_routing_table(data)
    assert len(targets) == 2
    assert targets[0].node_id == "gpu-1"
    assert targets[0].models == ("llama3.1:8b", "mistral:7b")
    assert targets[1].backend_auth == "Bearer sk-test"


def test_parse_empty():
    assert parse_fleet_routing_table({}) == []
    assert parse_fleet_routing_table({"fleet_routing": []}) == []


def test_parse_invalid_entry():
    """Invalid entries should be skipped, valid ones kept."""
    data = {
        "fleet_routing": [
            {"node_id": "good", "backend_url": "http://localhost:11434", "models": ["test"]},
            {"broken": True},  # Missing required fields
            {"node_id": "also-good", "backend_url": "http://localhost:8000"},
        ]
    }
    targets = parse_fleet_routing_table(data)
    assert len(targets) == 2
    assert targets[0].node_id == "good"
    assert targets[1].node_id == "also-good"
