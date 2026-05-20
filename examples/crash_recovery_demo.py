#!/usr/bin/env python3
"""Demo: Crash-Resilient Multi-Agent Task Execution

Demonstrates entropy-state's crash recovery:
1. Three "agents" (simulated) work on tasks
2. Agent 2 crashes mid-task
3. Agent 2 restarts and recovers state from peers
4. All tasks complete without data loss

This works WITHOUT an LLM — it simulates agent behavior
to demonstrate the infrastructure-level crash recovery.

Usage:
    python3 examples/crash_recovery_demo.py
"""

import asyncio
import sys
import time
import random

sys.path.insert(0, "/home/dweyh/entropy-state/python")

from entropy_state import AgentStateStore
from entropy_state.types import PeerAddress


async def simulate_agent(
    name: str,
    store: AgentStateStore,
    tasks: list,
    crash_after: int = 0,  # 0 = no crash
):
    """Simulate an agent working on tasks.

    Each task takes ~0.3s. If crash_after > 0, agent "crashes"
    after that many tasks (simulated by stopping early).
    """
    completed = 0

    for i, task in enumerate(tasks):
        # Check if task already done (recovered from peers)
        status = store.get(f"task/{name}/{i}/status")
        if status == "done":
            result = store.get(f"task/{name}/{i}/result")
            print(f"  🔄 {name}: Task {i} already done (recovered): {result}")
            completed += 1
            continue

        # Crash simulation
        if crash_after > 0 and completed >= crash_after:
            print(f"  💥 {name}: CRASHED after {completed} tasks!")
            store.set(f"agent/{name}/status", "crashed")
            return completed

        # Work on task
        store.set(f"task/{name}/{i}/status", "running")
        store.set(f"agent/{name}/status", "working")
        await asyncio.sleep(0.3 + random.random() * 0.2)  # simulate work

        # Complete task
        result = f"Result-{name}-{i}-{random.randint(100, 999)}"
        store.set(f"task/{name}/{i}/status", "done")
        store.set(f"task/{name}/{i}/result", result)
        print(f"  ✅ {name}: Task {i} done → {result}")
        completed += 1

    store.set(f"agent/{name}/status", "idle")
    return completed


async def run_demo():
    print("=" * 60)
    print("  🔬 Crash-Resilient Multi-Agent Demo")
    print("  Powered by entropy-state CRDT + Gossip")
    print("=" * 60)

    # Peer addresses
    p1 = PeerAddress("127.0.0.1", 18001)
    p2 = PeerAddress("127.0.0.1", 18002)
    p3 = PeerAddress("127.0.0.1", 18003)

    # Define tasks for each agent
    tasks_researcher = ["Find papers", "Summarize findings", "Extract data"]
    tasks_analyst = ["Analyze trends", "Build model", "Generate report"]
    tasks_writer = ["Draft intro", "Write body", "Format output"]

    # ─── Act 1: Start all agents ────────────────────────────────

    print("\n📍 Act 1: All agents start working")
    print("-" * 40)

    s1 = AgentStateStore(room="demo", node_id=1, port=18001, gossip_interval_ms=100)
    s2 = AgentStateStore(room="demo", node_id=2, port=18002, gossip_interval_ms=100)
    s3 = AgentStateStore(room="demo", node_id=3, port=18003, gossip_interval_ms=100)

    await s1.start(peers=[p2, p3])
    await s2.start(peers=[p1, p3])
    await s3.start(peers=[p1, p2])

    # Agent 2 will crash after 1 task
    r1 = await simulate_agent("researcher", s1, tasks_researcher)
    r2 = await simulate_agent("analyst", s2, tasks_analyst, crash_after=1)
    r3 = await simulate_agent("writer", s3, tasks_writer)

    # Wait for gossip to sync crash state
    await asyncio.sleep(0.5)

    # ─── Act 2: Verify crash state ──────────────────────────────

    print(f"\n📍 Act 2: Verify state after crash")
    print("-" * 40)

    # Check from agent 3's perspective (should have all state via gossip)
    analyst_status = s3.get("agent/analyst/status")
    print(f"  Agent 3 sees analyst status: {analyst_status}")
    task_0_status = s3.get("task/analyst/0/status")
    task_1_status = s3.get("task/analyst/1/status")
    print(f"  Agent 3 sees analyst task 0: {task_0_status}")
    print(f"  Agent 3 sees analyst task 1: {task_1_status or 'NOT STARTED (lost in crash)'}")

    # Stop crashed agent
    await s2.stop()

    # ─── Act 3: Restart agent 2 — state recovery ────────────────

    print(f"\n📍 Act 3: Analyst restarts — recovering from peers")
    print("-" * 40)

    # Create new store for recovered agent
    s2_recovered = AgentStateStore(
        room="demo", node_id=4, port=18002,  # new node_id, same port
        gossip_interval_ms=100,
    )
    await s2_recovered.start(peers=[p1, p3])

    # Wait for state recovery via gossip
    await asyncio.sleep(1.0)

    # Resume work — previously completed tasks are skipped
    r2_continued = await simulate_agent("analyst", s2_recovered, tasks_analyst)

    # Wait for final sync
    await asyncio.sleep(0.5)

    # ─── Act 4: Verify full completion ──────────────────────────

    print(f"\n📍 Act 4: Verify all tasks completed")
    print("-" * 40)

    all_done = True
    for agent_name, task_list in [
        ("researcher", tasks_researcher),
        ("analyst", tasks_analyst),
        ("writer", tasks_writer),
    ]:
        for i in range(len(task_list)):
            status = s1.get(f"task/{agent_name}/{i}/status")
            result = s1.get(f"task/{agent_name}/{i}/result")
            marker = "✅" if status == "done" else "❌"
            print(f"  {marker} {agent_name}/task/{i}: {status or 'missing'} → {result or 'N/A'}")
            if status != "done":
                all_done = False

    # ─── Stats ──────────────────────────────────────────────────

    print(f"\n📊 Gossip Stats:")
    stats = s1.stats
    print(f"  Entropy: {stats.entropy:.4f}")
    print(f"  Gossip rounds: {stats.gossip_rounds}")
    print(f"  Keys synced: {stats.key_count}")
    print(f"  Peers: {stats.peer_count}")

    # Cleanup
    await s1.stop()
    await s2_recovered.stop()
    await s3.stop()

    if all_done:
        print(f"\n{'=' * 60}")
        print(f"  🎉 SUCCESS: Zero data loss after agent crash!")
        print(f"  All 9 tasks completed across 3 agents.")
        print(f"  Agent 'analyst' crashed after 1 task, recovered 1, completed 2 more.")
        print(f"{'=' * 60}")
    else:
        print(f"\n  ⚠️ Some tasks not completed — check output above.")

    return all_done


if __name__ == "__main__":
    success = asyncio.run(run_demo())
    sys.exit(0 if success else 1)
