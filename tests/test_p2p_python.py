"""P2P Gossip Integration Test — Pure Python SDK.

Tests two AgentStateStore instances syncing via TCP gossip
WITHOUT any Rust binary. Pure Python CRDT + Push-Pull.
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from propagul.store import AgentStateStore
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


async def run_tests():
    print("=" * 60)
    print("P2P GOSSIP INTEGRATION TEST (Pure Python)")
    print("=" * 60)

    # ─── Test 1: Two-node sync ─────────────────────────────────
    print("\n--- Two-Node Sync ---")

    store1 = AgentStateStore(room="test", node_id=1, port=19001)
    store2 = AgentStateStore(room="test", node_id=2, port=19002)

    # Set data before starting gossip
    store1.set("from_1", "hello")
    store2.set("from_2", "world")

    # Start both with each other as peers
    await store1.start(peers=[PeerAddress("127.0.0.1", 19002)])
    await store2.start(peers=[PeerAddress("127.0.0.1", 19001)])

    # Wait for gossip to sync (a few rounds)
    await asyncio.sleep(2.0)

    test("store1 has store2's data", store1.get("from_2") == "world",
         f"got: {store1.get('from_2')}")
    test("store2 has store1's data", store2.get("from_1") == "hello",
         f"got: {store2.get('from_1')}")

    # ─── Test 2: Live update propagation ──────────────────────
    print("\n--- Live Update Propagation ---")

    store1.set("live_key", "updated_value")
    await asyncio.sleep(1.5)

    test("live update propagated", store2.get("live_key") == "updated_value",
         f"got: {store2.get('live_key')}")

    # ─── Test 3: Delete propagation ───────────────────────────
    print("\n--- Delete Propagation ---")

    store1.delete("from_1")
    await asyncio.sleep(1.5)

    test("delete propagated", store2.get("from_1") is None,
         f"got: {store2.get('from_1')}")

    # ─── Test 4: Change callbacks ──────────────────────────────
    print("\n--- Change Callbacks ---")

    changes = []
    store2.on_change(lambda k, v: changes.append((k, v)))

    store1.set("callback_test", "trigger")
    await asyncio.sleep(1.5)

    test("change callback fired", len(changes) > 0, f"changes: {changes}")
    if changes:
        test("callback has correct key", changes[-1][0] == "callback_test")
        test("callback has correct value", changes[-1][1] == "trigger")

    # ─── Test 5: Stats ─────────────────────────────────────────
    print("\n--- Stats ---")

    s1 = store1.stats
    s2 = store2.stats

    test("store1 gossip rounds > 0", s1.gossip_rounds > 0,
         f"rounds: {s1.gossip_rounds}")
    test("store2 gossip rounds > 0", s2.gossip_rounds > 0,
         f"rounds: {s2.gossip_rounds}")
    test("store1 peers = 1", s1.peer_count == 1, f"peers: {s1.peer_count}")

    # ─── Test 6: Three-node sync ───────────────────────────────
    print("\n--- Three-Node Sync ---")

    store3 = AgentStateStore(room="test", node_id=3, port=19003)
    store3.set("from_3", "third_node")

    await store3.start(peers=[
        PeerAddress("127.0.0.1", 19001),
        PeerAddress("127.0.0.1", 19002),
    ])

    # k=1 with 3 nodes: each node sends to 1 random peer per round
    # Store1 has 2 peers, 50% chance per round to pick Store3
    # After ~10 rounds (500ms each = 5s), probability of NEVER
    # picking Store3 = 0.5^10 = 0.1%. So 8s should be sufficient.
    await asyncio.sleep(8.0)

    test("store3 has store1 data", store3.get("live_key") == "updated_value",
         f"got: {store3.get('live_key')}")
    test("store1 has store3 data", store1.get("from_3") == "third_node",
         f"got: {store1.get('from_3')}")

    # ─── Cleanup ──────────────────────────────────────────────
    await store1.stop()
    await store2.stop()
    await store3.stop()

    print(f"\n{'=' * 60}")
    print(f"P2P GOSSIP: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
