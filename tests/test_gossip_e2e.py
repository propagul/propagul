#!/usr/bin/env python3
"""Integration test: 3 agents sync state via TCP gossip.

Verifies the full stack: AgentStateStore → EntropyAgent → OR-Map CRDT → TCP Transport.

Test scenario:
1. Start 3 agents on different ports
2. Agent 1 sets key "task" = "research"
3. Agent 2 sets key "status" = "running"
4. Wait for gossip convergence
5. Verify all agents have both keys
6. Agent 3 deletes "task"
7. Verify deletion propagates
"""

import asyncio
import sys
import time

# Add parent directory to path for development
sys.path.insert(0, "/home/dweyh/entropy-state/python")

from propagul import AgentStateStore, StaticDiscovery
from propagul.types import PeerAddress


async def run_test():
    passed = 0
    failed = 0

    def test(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name} — {detail}")

    # Peer addresses
    p1 = PeerAddress("127.0.0.1", 19001)
    p2 = PeerAddress("127.0.0.1", 19002)
    p3 = PeerAddress("127.0.0.1", 19003)

    # Create stores
    s1 = AgentStateStore(room="test", node_id=1, port=19001, gossip_interval_ms=100)
    s2 = AgentStateStore(room="test", node_id=2, port=19002, gossip_interval_ms=100)
    s3 = AgentStateStore(room="test", node_id=3, port=19003, gossip_interval_ms=100)

    print("=== Phase 1: Start ===")

    await s1.start(peers=[p2, p3])
    await s2.start(peers=[p1, p3])
    await s3.start(peers=[p1, p2])

    test("all running", s1.is_running and s2.is_running and s3.is_running)

    print("\n=== Phase 2: Set state ===")

    s1.set("task", "research")
    s2.set("status", "running")
    s3.set("agent", "agent-3")

    # Wait for gossip rounds
    await asyncio.sleep(1.5)

    print("\n=== Phase 3: Verify convergence ===")

    # Each store should have all 3 keys
    test("s1 has task", s1.get("task") == "research", f"got {s1.get('task')}")
    test("s1 has status", s1.get("status") == "running", f"got {s1.get('status')}")
    test("s1 has agent", s1.get("agent") == "agent-3", f"got {s1.get('agent')}")

    test("s2 has task", s2.get("task") == "research", f"got {s2.get('task')}")
    test("s2 has status", s2.get("status") == "running", f"got {s2.get('status')}")
    test("s2 has agent", s2.get("agent") == "agent-3", f"got {s2.get('agent')}")

    test("s3 has task", s3.get("task") == "research", f"got {s3.get('task')}")
    test("s3 has status", s3.get("status") == "running", f"got {s3.get('status')}")
    test("s3 has agent", s3.get("agent") == "agent-3", f"got {s3.get('agent')}")

    print("\n=== Phase 4: Delete propagation ===")

    s3.delete("task")
    await asyncio.sleep(1.0)

    test("s1: task deleted", s1.get("task") is None, f"got {s1.get('task')}")
    test("s2: task deleted", s2.get("task") is None, f"got {s2.get('task')}")
    test("s3: task deleted", s3.get("task") is None, f"got {s3.get('task')}")

    print("\n=== Phase 5: Stats ===")

    stats = s1.stats
    test("stats.entropy >= 0", stats.entropy >= 0, f"entropy={stats.entropy}")
    test("stats.key_count > 0", stats.key_count > 0, f"keys={stats.key_count}")
    test("stats.peer_count == 2", stats.peer_count == 2, f"peers={stats.peer_count}")
    test("stats.gossip_rounds > 0", stats.gossip_rounds > 0, f"rounds={stats.gossip_rounds}")
    print(f"  📊 entropy={stats.entropy:.4f} k={stats.current_k} "
          f"loss={stats.loss_rate:.2%} rounds={stats.gossip_rounds}")

    print("\n=== Phase 6: Change callback ===")

    changes = []
    s1.on_change(lambda key, val: changes.append((key, val)))
    s2.set("new_key", "hello")
    await asyncio.sleep(1.0)
    test("change callback fired", len(changes) > 0, f"changes={changes}")
    if changes:
        test("correct change key", changes[0][0] == "new_key", f"key={changes[0][0]}")

    print("\n=== Phase 6b: Deletion callback ===")

    del_changes = []
    s2.on_change(lambda key, val: del_changes.append((key, val)))
    s1.delete("new_key")
    await asyncio.sleep(1.0)
    del_events = [(k, v) for k, v in del_changes if k == "new_key" and v is None]
    test("deletion callback fires with None", len(del_events) > 0, f"del_changes={del_changes}")

    print("\n=== Phase 7: Concurrent writes ===")

    s1.set("color", "red")
    s2.set("color", "blue")
    await asyncio.sleep(1.0)
    conflicts_1 = s1.get_conflicts("color")
    test("concurrent conflicts detected", len(conflicts_1) >= 1, f"conflicts={conflicts_1}")

    print("\n=== Cleanup ===")

    await s1.stop()
    await s2.stop()
    await s3.stop()

    test("all stopped", not s1.is_running and not s2.is_running and not s3.is_running)

    print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)
