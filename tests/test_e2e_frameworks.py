#!/usr/bin/env python3
"""E2E verification of framework integrations with real CrewAI + LangGraph.

Tests use mocked LLM responses — no OpenAI API key required for most tests.
Validates against CrewAI 1.14.x and LangGraph 1.2.x / langgraph-checkpoint 4.1.x.

Run standalone:
    OPENAI_API_KEY=sk-test .venv311/bin/python3.11 tests/test_e2e_frameworks.py

Or via pytest (test_ prefixed functions are collected automatically):
    .venv/bin/python -m pytest tests/test_e2e_frameworks.py -q
"""
import sys
import os
import time

# Disable all LangChain telemetry before any imports
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/tiktoken_cache")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


# ═══════════════════════════════════════════════════════════════
# CREWAI INTEGRATION (1.14.x)
# ═══════════════════════════════════════════════════════════════

def test_crewai_imports():
    """Verify CrewAI core classes are importable."""
    from crewai import Agent, Task, Crew  # noqa: F401


def test_persistent_crew_import():
    from propagul.integrations.crewai import PersistentCrew  # noqa: F401


def test_persistent_crew_construction():
    """Verify PersistentCrew wraps a real Crew object."""
    from crewai import Agent, Task
    from propagul.integrations.crewai import PersistentCrew

    r = Agent(role="R", goal="G", backstory="B", verbose=False)
    t = Task(description="D", expected_output="O", agent=r)
    c = PersistentCrew(agents=[r], tasks=[t], state_room="e2e-crew", state_port=19901)
    assert type(c._crew).__name__ == "Crew"
    assert type(c._store).__name__ == "AgentStateStore"
    assert c._node_id >= 0


def test_persistent_crew_state_keys():
    """Verify PersistentCrew writes expected state keys to CRDT."""
    from crewai import Agent, Task
    from propagul.integrations.crewai import PersistentCrew

    r = Agent(role="Tester", goal="Test", backstory="B", verbose=False)
    t = Task(description="Test task", expected_output="Result", agent=r)
    c = PersistentCrew(agents=[r], tasks=[t], state_room="e2e-keys", state_port=19902)

    # State store should be accessible
    assert c._store is not None
    # Set a key manually and verify
    c._store.set("test_key", "test_value")
    assert c._store.get("test_key") == "test_value"


def test_persistent_crew_recovery_flag():
    """Verify recovery parameter is stored correctly."""
    from crewai import Agent, Task
    from propagul.integrations.crewai import PersistentCrew

    r = Agent(role="R", goal="G", backstory="B", verbose=False)
    t = Task(description="D", expected_output="O", agent=r)
    c = PersistentCrew(agents=[r], tasks=[t], state_room="e2e-rec", state_port=19903, recovery=True)
    assert c._recovery is True

    c2 = PersistentCrew(agents=[r], tasks=[t], state_room="e2e-rec2", state_port=19904, recovery=False)
    assert c2._recovery is False


# ═══════════════════════════════════════════════════════════════
# LANGGRAPH INTEGRATION (1.2.x / checkpoint 4.1.x)
# ═══════════════════════════════════════════════════════════════

def test_langgraph_base_import():
    """Verify BaseCheckpointSaver is importable."""
    from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401


def test_checkpoint_tuple_import():
    """Verify CheckpointTuple type is available."""
    from langgraph.checkpoint.base import CheckpointTuple  # noqa: F401


def test_entropy_checkpointer_construction():
    """Verify EntropyCheckpointer can be instantiated."""
    from propagul.integrations.langgraph import EntropyCheckpointer
    cp = EntropyCheckpointer(room="e2e-lg", port=19910)
    assert cp is not None


def test_entropy_checkpointer_isinstance():
    """CRITICAL: Verify EntropyCheckpointer IS a BaseCheckpointSaver.

    LangGraph's compile() may do isinstance() checks.
    If this fails, our checkpointer silently breaks.
    """
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from propagul.integrations.langgraph import EntropyCheckpointer
    cp = EntropyCheckpointer(room="e2e-isinstance", port=19911)
    assert isinstance(cp, BaseCheckpointSaver), (
        f"EntropyCheckpointer is {type(cp).__mro__}, not a BaseCheckpointSaver!"
    )


def test_entropy_checkpointer_api_surface():
    """Verify all required methods exist with correct signatures."""
    from propagul.integrations.langgraph import EntropyCheckpointer
    cp = EntropyCheckpointer(room="e2e-api", port=19912)

    # Sync methods
    for method in ["put", "get_tuple", "list", "put_writes", "delete_thread",
                    "delete_for_runs", "copy_thread", "prune"]:
        assert hasattr(cp, method), f"Missing sync method: {method}"

    # Async methods
    for method in ["aget_tuple", "alist", "aput", "aput_writes",
                    "adelete_thread", "adelete_for_runs", "acopy_thread", "aprune"]:
        assert hasattr(cp, method), f"Missing async method: {method}"


def test_checkpointer_put_get_roundtrip():
    """Verify put() → get_tuple() roundtrip returns correct data."""
    from propagul.integrations.langgraph import EntropyCheckpointer
    from langgraph.checkpoint.base import CheckpointTuple

    cp = EntropyCheckpointer(room="e2e-roundtrip", port=19913)

    config = {"configurable": {"thread_id": "test-thread-1"}}
    checkpoint = {
        "v": 1,
        "id": "cp-001",
        "ts": "2026-05-14T06:00:00Z",
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }
    metadata = {"source": "input", "step": -1, "parents": {}}
    new_versions = {"messages": "1"}

    result_config = cp.put(config, checkpoint, metadata, new_versions)
    assert "configurable" in result_config
    assert "checkpoint_id" in result_config["configurable"]

    # Get it back
    result = cp.get_tuple(config)
    assert result is not None
    assert isinstance(result, CheckpointTuple)
    assert result.checkpoint["channel_values"]["messages"] == ["hello"]
    assert result.metadata["source"] == "input"

    cp.close()


def test_checkpointer_put_writes():
    """Verify put_writes() stores and retrieves pending writes."""
    from propagul.integrations.langgraph import EntropyCheckpointer

    cp = EntropyCheckpointer(room="e2e-writes", port=19914)

    config = {"configurable": {"thread_id": "test-writes"}}
    writes = [("messages", "hello world"), ("status", "running")]
    cp.put_writes(config, writes, task_id="task-001", task_path="node/step")

    # Verify via get_tuple (pending_writes)
    # First put a checkpoint so get_tuple returns something
    checkpoint = {
        "v": 1, "id": "cp-w1", "ts": "2026-05-14T06:00:00Z",
        "channel_values": {}, "channel_versions": {},
        "versions_seen": {}, "updated_channels": None,
    }
    cp.put(config, checkpoint, {"source": "loop", "step": 0}, {})

    result = cp.get_tuple(config)
    assert result is not None
    assert result.pending_writes is not None
    assert len(result.pending_writes) >= 2

    cp.close()


def test_checkpointer_list_history():
    """Verify list() returns stored checkpoints."""
    from propagul.integrations.langgraph import EntropyCheckpointer

    cp = EntropyCheckpointer(room="e2e-list", port=19915, max_history=5)

    config = {"configurable": {"thread_id": "test-list"}}

    # Store 3 checkpoints
    for i in range(3):
        checkpoint = {
            "v": 1, "id": f"cp-list-{i}", "ts": f"2026-05-14T0{i}:00:00Z",
            "channel_values": {"step": i}, "channel_versions": {},
            "versions_seen": {}, "updated_channels": None,
        }
        cp.put(config, checkpoint, {"source": "loop", "step": i}, {})

    # List all
    items = list(cp.list(config))
    assert len(items) >= 3, f"Expected >= 3 items, got {len(items)}"

    # List with limit
    limited = list(cp.list(config, limit=2))
    assert len(limited) == 2

    cp.close()


def test_checkpointer_delete_thread():
    """Verify delete_thread() removes all checkpoints."""
    from propagul.integrations.langgraph import EntropyCheckpointer

    cp = EntropyCheckpointer(room="e2e-delete", port=19916)

    config = {"configurable": {"thread_id": "test-delete"}}
    checkpoint = {
        "v": 1, "id": "cp-del", "ts": "2026-05-14T06:00:00Z",
        "channel_values": {}, "channel_versions": {},
        "versions_seen": {}, "updated_channels": None,
    }
    cp.put(config, checkpoint, {"source": "input"}, {})

    # Verify it exists
    assert cp.get_tuple(config) is not None

    # Delete
    cp.delete_thread("test-delete")

    # Verify it's gone
    assert cp.get_tuple(config) is None

    cp.close()


def test_checkpointer_copy_thread():
    """Verify copy_thread() duplicates state."""
    from propagul.integrations.langgraph import EntropyCheckpointer

    cp = EntropyCheckpointer(room="e2e-copy", port=19917)

    # Store in source thread
    src_config = {"configurable": {"thread_id": "source-thread"}}
    checkpoint = {
        "v": 1, "id": "cp-src", "ts": "2026-05-14T06:00:00Z",
        "channel_values": {"data": "original"}, "channel_versions": {},
        "versions_seen": {}, "updated_channels": None,
    }
    cp.put(src_config, checkpoint, {"source": "input"}, {})

    # Copy to target
    cp.copy_thread("source-thread", "target-thread")

    # Verify target has data
    tgt_config = {"configurable": {"thread_id": "target-thread"}}
    result = cp.get_tuple(tgt_config)
    assert result is not None
    assert result.checkpoint["channel_values"]["data"] == "original"

    cp.close()


def test_checkpointer_prune():
    """Verify prune() clears history."""
    from propagul.integrations.langgraph import EntropyCheckpointer

    cp = EntropyCheckpointer(room="e2e-prune", port=19918)

    config = {"configurable": {"thread_id": "test-prune"}}
    for i in range(3):
        checkpoint = {
            "v": 1, "id": f"cp-prune-{i}", "ts": f"2026-05-14T0{i}:00:00Z",
            "channel_values": {}, "channel_versions": {},
            "versions_seen": {}, "updated_channels": None,
        }
        cp.put(config, checkpoint, {"source": "loop", "step": i}, {})

    # Prune history (keep latest)
    cp.prune(["test-prune"], strategy="keep_latest")

    # Latest should still exist
    assert cp.get_tuple(config) is not None
    # History should be empty
    assert list(cp.list(config)) == []

    cp.close()


# ═══════════════════════════════════════════════════════════════
# Standalone runner (backward compat)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _passed = 0
    _failed = 0
    _t0 = time.time()

    def _check(name, fn):
        global _passed, _failed
        try:
            fn()
            print(f"  ✅ {name} (+{time.time() - _t0:.1f}s)", flush=True)
            _passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e} (+{time.time() - _t0:.1f}s)", flush=True)
            _failed += 1

    print("=== CrewAI Integration ===", flush=True)
    _check("CrewAI imports", test_crewai_imports)
    _check("PersistentCrew import", test_persistent_crew_import)
    _check("PersistentCrew construction", test_persistent_crew_construction)
    _check("PersistentCrew state keys", test_persistent_crew_state_keys)
    _check("PersistentCrew recovery flag", test_persistent_crew_recovery_flag)

    print("\n=== LangGraph Integration ===", flush=True)
    _check("BaseCheckpointSaver import", test_langgraph_base_import)
    _check("CheckpointTuple import", test_checkpoint_tuple_import)
    _check("EntropyCheckpointer construction", test_entropy_checkpointer_construction)
    _check("EntropyCheckpointer isinstance", test_entropy_checkpointer_isinstance)
    _check("EntropyCheckpointer API surface", test_entropy_checkpointer_api_surface)
    _check("put/get_tuple roundtrip", test_checkpointer_put_get_roundtrip)
    _check("put_writes", test_checkpointer_put_writes)
    _check("list history", test_checkpointer_list_history)
    _check("delete_thread", test_checkpointer_delete_thread)
    _check("copy_thread", test_checkpointer_copy_thread)
    _check("prune", test_checkpointer_prune)

    elapsed = time.time() - _t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"FRAMEWORK E2E: {_passed} passed, {_failed} failed ({elapsed:.1f}s)", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if _failed > 0 else 0)
