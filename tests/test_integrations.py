#!/usr/bin/env python3
"""Tests for Phase 3: Integrations + Demo validation."""

import asyncio
import json
import sys
import time

sys.path.insert(0, "/home/dweyh/entropy-state/python")

from propagul import AgentStateStore
from propagul.types import PeerAddress

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


# ─── LangGraph Checkpointer Tests ───────────────────────────────

print("=== LangGraph Checkpointer Tests ===")

from propagul.integrations.langgraph import EntropyCheckpointer


async def test_checkpointer():
    # Use unique ports to avoid conflicts
    cp = EntropyCheckpointer(room="test-lg", port=17001, node_id=99)
    await cp.astart()

    config = {"configurable": {"thread_id": "thread-1"}}
    checkpoint = {"channel_values": {"messages": ["hello"]}, "v": 1}
    metadata = {"source": "test", "step": 0}

    # put
    result_config = cp.put(config, checkpoint, metadata)
    test("put returns config", "thread_id" in result_config.get("configurable", {}))
    test("put returns checkpoint_id",
         "checkpoint_id" in result_config.get("configurable", {}))

    # get_tuple
    retrieved = cp.get_tuple(config)
    test("get_tuple returns result", retrieved is not None)
    if retrieved:
        r_config, r_checkpoint, r_meta = retrieved
        test("get_tuple: checkpoint matches",
             r_checkpoint.get("v") == 1, f"got {r_checkpoint}")
        test("get_tuple: metadata matches",
             r_meta.get("source") == "test", f"got {r_meta}")

    # put_writes
    writes = [("messages", "world"), ("counter", 42)]
    cp.put_writes(config, writes, task_id="task-abc")
    # No crash = success
    test("put_writes: no crash", True)

    # list
    history = list(cp.list(config))
    test("list: has history", len(history) > 0, f"len={len(history)}")

    # Multiple checkpoints
    for i in range(3):
        cp.put(config, {"v": i + 2}, {"step": i + 1})

    history2 = list(cp.list(config))
    test("list: multiple checkpoints", len(history2) >= 3, f"len={len(history2)}")

    # Latest should be most recent
    latest = cp.get_tuple(config)
    test("get_tuple: latest is most recent",
         latest is not None and latest[1].get("v") == 4, f"got {latest}")

    # delete_thread
    cp.delete_thread("thread-1")
    deleted = cp.get_tuple(config)
    test("delete_thread: clears data", deleted is None, f"got {deleted}")

    await cp.aclose()


asyncio.run(test_checkpointer())


# ─── CrewAI Plugin Import Tests ─────────────────────────────────

print("\n=== CrewAI Plugin Tests ===")

# Test lazy import (should not crash even without crewai)
try:
    from propagul.integrations.crewai import PersistentCrew, _check_crewai
    test("crewai module imports", True)
    test("crewai availability check", isinstance(_check_crewai(), bool))

    # If crewai is available, test construction
    if _check_crewai():
        print("  ℹ️  CrewAI is available — skipping full test (requires LLM)")
    else:
        # Verify error message when crewai not installed
        try:
            PersistentCrew(agents=[], tasks=[])
            test("crewai missing: raises ImportError", False, "should have raised")
        except ImportError as e:
            test("crewai missing: raises ImportError", "pip install" in str(e))
except Exception as e:
    test("crewai module imports", False, str(e))


# ─── Crash Recovery Test ────────────────────────────────────────

print("\n=== Crash Recovery Tests ===")


async def test_crash_recovery():
    p1 = PeerAddress("127.0.0.1", 17011)
    p2 = PeerAddress("127.0.0.1", 17012)

    s1 = AgentStateStore(room="cr-test", node_id=1, port=17011, gossip_interval_ms=100)
    s2 = AgentStateStore(room="cr-test", node_id=2, port=17012, gossip_interval_ms=100)

    await s1.start(peers=[p2])
    await s2.start(peers=[p1])

    # Agent 1 sets state
    s1.set("task/0/status", "done")
    s1.set("task/0/result", "research-complete")
    s1.set("task/1/status", "running")

    await asyncio.sleep(0.5)

    # Verify s2 has state
    test("pre-crash: s2 has task 0", s2.get("task/0/status") == "done")

    # "Crash" s2
    await s2.stop()

    # Agent 1 continues
    s1.set("task/1/status", "done")
    s1.set("task/1/result", "analysis-complete")

    # "Restart" s2 (new node_id, same port)
    s2_new = AgentStateStore(room="cr-test", node_id=3, port=17012, gossip_interval_ms=100)
    await s2_new.start(peers=[p1])
    await asyncio.sleep(1.0)

    # Verify recovery
    test("post-recovery: s2 has task 0",
         s2_new.get("task/0/status") == "done")
    test("post-recovery: s2 has task 0 result",
         s2_new.get("task/0/result") == "research-complete")
    test("post-recovery: s2 has task 1",
         s2_new.get("task/1/status") == "done")
    test("post-recovery: s2 has task 1 result",
         s2_new.get("task/1/result") == "analysis-complete")

    await s1.stop()
    await s2_new.stop()


asyncio.run(test_crash_recovery())


# ─── Summary ────────────────────────────────────────────────────

print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
