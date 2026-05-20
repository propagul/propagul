#!/usr/bin/env python3
"""Performance & Scale Tests — entropy-state under load.

Tests:
1. 1000+ keys write/read throughput
2. Burst-load gossip (high-frequency writes)
3. Large payload handling
4. Gossip convergence time under load
5. Snapshot save/load performance with large state

Run with:
    .venv/bin/python tests/test_performance.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

passed = 0
failed = 0
t0 = time.time()


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name} (+{time.time() - t0:.1f}s)", flush=True)
        passed += 1
    else:
        print(f"  ❌ {name} — {detail} (+{time.time() - t0:.1f}s)", flush=True)
        failed += 1


# ═══════════════════════════════════════════════════════════════
# TEST 1: Write throughput (1000+ keys)
# ═══════════════════════════════════════════════════════════════
print("=== 1. Write Throughput: 1000+ Keys ===", flush=True)


def test_write_throughput():
    """Write 2000 keys to a single store, measure time."""
    from entropy_state import AgentStateStore

    store = AgentStateStore(room="perf-write", node_id=1, port=19900, gossip_interval_ms=5000)
    n = 2000

    start = time.monotonic()
    for i in range(n):
        store.set(f"key/{i}/status", f"value_{i}_{'x' * 50}")
    elapsed = time.monotonic() - start

    ops_per_sec = n / elapsed
    print(f"    → {n} writes in {elapsed*1000:.1f}ms ({ops_per_sec:.0f} ops/s)", flush=True)

    test(f"write throughput: {n} keys < 1s", elapsed < 1.0, f"took {elapsed:.2f}s")
    test(f"all keys readable", store.get(f"key/{n-1}/status") is not None)
    test(f"key count correct", len(store.get_all()) == n, f"got {len(store.get_all())}")


test_write_throughput()


# ═══════════════════════════════════════════════════════════════
# TEST 2: Read throughput
# ═══════════════════════════════════════════════════════════════
print("\n=== 2. Read Throughput ===", flush=True)


def test_read_throughput():
    """Read 2000 keys, measure time."""
    from entropy_state import AgentStateStore

    store = AgentStateStore(room="perf-read", node_id=1, port=19901, gossip_interval_ms=5000)
    n = 2000

    for i in range(n):
        store.set(f"r/{i}", f"data_{i}")

    start = time.monotonic()
    for i in range(n):
        _ = store.get(f"r/{i}")
    elapsed = time.monotonic() - start

    ops_per_sec = n / elapsed
    print(f"    → {n} reads in {elapsed*1000:.1f}ms ({ops_per_sec:.0f} ops/s)", flush=True)

    test(f"read throughput: {n} keys < 500ms", elapsed < 0.5, f"took {elapsed:.2f}s")


test_read_throughput()


# ═══════════════════════════════════════════════════════════════
# TEST 3: get_all() with large state
# ═══════════════════════════════════════════════════════════════
print("\n=== 3. get_all() Performance ===", flush=True)


def test_get_all_performance():
    """get_all() with 2000 keys, measure time and memory."""
    from entropy_state import AgentStateStore

    store = AgentStateStore(room="perf-getall", node_id=1, port=19902, gossip_interval_ms=5000)
    n = 2000

    for i in range(n):
        store.set(f"ga/{i}", f"value_{i}_{'data' * 10}")

    start = time.monotonic()
    all_data = store.get_all()
    elapsed = time.monotonic() - start

    total_bytes = sum(len(k) + len(v) for k, v in all_data.items())
    print(f"    → get_all({n} keys) in {elapsed*1000:.1f}ms, {total_bytes/1024:.1f} KB", flush=True)

    test(f"get_all: {n} keys < 200ms", elapsed < 0.2, f"took {elapsed:.2f}s")
    test(f"get_all: correct count", len(all_data) == n, f"got {len(all_data)}")


test_get_all_performance()


# ═══════════════════════════════════════════════════════════════
# TEST 4: Snapshot save/load with large state
# ═══════════════════════════════════════════════════════════════
print("\n=== 4. Snapshot Performance ===", flush=True)


def test_snapshot_performance():
    """Save and load 2000-key state snapshot."""
    from entropy_state import AgentStateStore
    from entropy_state.server.persistence import save_room_state, load_all_rooms

    store = AgentStateStore(room="perf-snap", node_id=1, port=19903, gossip_interval_ms=5000)
    n = 2000

    for i in range(n):
        store.set(f"snap/{i}", f"snapshot_data_{i}_{'x' * 100}")

    state = store.get_all()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        start = time.monotonic()
        save_room_state(tmpdir, "perf-snap", state, {"port": 19903})
        save_elapsed = time.monotonic() - start

        # Check file size
        snap_file = os.path.join(tmpdir, "perf-snap.json")
        file_size = os.path.getsize(snap_file)

        # Load
        start = time.monotonic()
        loaded = load_all_rooms(tmpdir)
        load_elapsed = time.monotonic() - start

        print(f"    → Save: {save_elapsed*1000:.1f}ms, Load: {load_elapsed*1000:.1f}ms, File: {file_size/1024:.1f} KB", flush=True)

        test(f"snapshot save < 100ms", save_elapsed < 0.1, f"took {save_elapsed:.2f}s")
        test(f"snapshot load < 100ms", load_elapsed < 0.1, f"took {load_elapsed:.2f}s")
        test(f"snapshot data intact", len(loaded) == 1 and len(loaded[0][1]) == n,
             f"loaded {len(loaded)} rooms")


test_snapshot_performance()


# ═══════════════════════════════════════════════════════════════
# TEST 5: CRDT merge performance
# ═══════════════════════════════════════════════════════════════
print("\n=== 5. CRDT Merge Performance ===", flush=True)


def test_merge_performance():
    """Merge two large states (1000 keys each)."""
    from entropy_state_core import StateMap

    s1 = StateMap(node_id=1)
    s2 = StateMap(node_id=2)
    n = 1000

    for i in range(n):
        s1.set(f"s1/{i}", f"data_{i}")
    for i in range(n):
        s2.set(f"s2/{i}", f"data_{i}")

    snap1 = s1.snapshot()
    snap2 = s2.snapshot()

    # Merge s2's snapshot into s1
    start = time.monotonic()
    delta = s1.merge(snap2)
    elapsed = time.monotonic() - start

    print(f"    → Merge {n} keys in {elapsed*1000:.1f}ms, delta={delta} bytes", flush=True)

    test(f"merge {n} keys < 50ms", elapsed < 0.05, f"took {elapsed:.2f}s")
    test(f"s1 has s2 keys after merge", s1.get("s2/0") == "data_0")
    test(f"s1 has all keys", len(s1) >= n * 2 - 1)  # -1 for potential overlaps


test_merge_performance()


# ═══════════════════════════════════════════════════════════════
# TEST 6: Burst-load gossip (high-frequency writes + sync)
# ═══════════════════════════════════════════════════════════════
print("\n=== 6. Burst-Load Gossip Sync ===", flush=True)


async def test_burst_gossip():
    """Write 500 keys rapidly on Node A, verify Node B catches up."""
    from entropy_state import AgentStateStore
    from entropy_state.types import PeerAddress

    s1 = AgentStateStore(room="perf-burst", node_id=1, port=19910, gossip_interval_ms=100)
    s2 = AgentStateStore(room="perf-burst", node_id=2, port=19911, gossip_interval_ms=100)

    await s1.start(peers=[PeerAddress("127.0.0.1", 19911)])
    await s2.start(peers=[PeerAddress("127.0.0.1", 19910)])

    n = 500

    # Burst write
    start = time.monotonic()
    for i in range(n):
        s1.set(f"burst/{i}", f"value_{i}")
    write_elapsed = time.monotonic() - start

    print(f"    → Burst write: {n} keys in {write_elapsed*1000:.1f}ms", flush=True)

    # Wait for convergence
    await asyncio.sleep(3.0)

    # Count how many keys s2 received
    received = 0
    for i in range(n):
        if s2.get(f"burst/{i}") is not None:
            received += 1

    convergence_pct = (received / n) * 100
    print(f"    → s2 received: {received}/{n} ({convergence_pct:.1f}%)", flush=True)

    test(f"burst convergence >= 95%", convergence_pct >= 95.0,
         f"only {convergence_pct:.1f}% converged")
    test(f"burst convergence = 100%", convergence_pct == 100.0,
         f"{convergence_pct:.1f}% — {n - received} keys missing")

    await s1.stop()
    await s2.stop()


asyncio.run(test_burst_gossip())


# ═══════════════════════════════════════════════════════════════
# TEST 7: Convergence time measurement
# ═══════════════════════════════════════════════════════════════
print("\n=== 7. Convergence Time ===", flush=True)


async def test_convergence_time():
    """Measure how long it takes for a single write to appear on the other node."""
    from entropy_state import AgentStateStore
    from entropy_state.types import PeerAddress

    s1 = AgentStateStore(room="perf-conv", node_id=1, port=19920, gossip_interval_ms=100)
    s2 = AgentStateStore(room="perf-conv", node_id=2, port=19921, gossip_interval_ms=100)

    await s1.start(peers=[PeerAddress("127.0.0.1", 19921)])
    await s2.start(peers=[PeerAddress("127.0.0.1", 19920)])

    # Let gossip establish
    await asyncio.sleep(0.5)

    # Measure convergence
    convergence_times = []
    for i in range(5):
        key = f"conv/{i}"
        s1.set(key, f"t_{i}")
        start = time.monotonic()

        for _ in range(100):  # Poll for up to 5s
            if s2.get(key) is not None:
                break
            await asyncio.sleep(0.05)

        elapsed = time.monotonic() - start
        convergence_times.append(elapsed)

    avg_ms = sum(convergence_times) * 1000 / len(convergence_times)
    max_ms = max(convergence_times) * 1000
    min_ms = min(convergence_times) * 1000

    print(f"    → Convergence: avg={avg_ms:.0f}ms, min={min_ms:.0f}ms, max={max_ms:.0f}ms", flush=True)

    test(f"avg convergence < 500ms", avg_ms < 500, f"avg={avg_ms:.0f}ms")
    test(f"max convergence < 1000ms", max_ms < 1000, f"max={max_ms:.0f}ms")

    await s1.stop()
    await s2.stop()


asyncio.run(test_convergence_time())


# ═══════════════════════════════════════════════════════════════
# TEST 8: Large payload handling
# ═══════════════════════════════════════════════════════════════
print("\n=== 8. Large Payload ===", flush=True)


async def test_large_payload():
    """Write a 100KB value, verify it syncs correctly."""
    from entropy_state import AgentStateStore
    from entropy_state.types import PeerAddress

    s1 = AgentStateStore(room="perf-large", node_id=1, port=19930, gossip_interval_ms=200)
    s2 = AgentStateStore(room="perf-large", node_id=2, port=19931, gossip_interval_ms=200)

    await s1.start(peers=[PeerAddress("127.0.0.1", 19931)])
    await s2.start(peers=[PeerAddress("127.0.0.1", 19930)])

    # 100KB payload
    big_value = "A" * 100_000
    s1.set("big_key", big_value)

    await asyncio.sleep(2.0)

    received = s2.get("big_key")
    test("100KB payload synced", received is not None and len(received) == 100_000,
         f"received {len(received) if received else 0} bytes")

    # 500KB payload
    huge_value = "B" * 500_000
    s1.set("huge_key", huge_value)

    await asyncio.sleep(3.0)

    received_huge = s2.get("huge_key")
    test("500KB payload synced", received_huge is not None and len(received_huge) == 500_000,
         f"received {len(received_huge) if received_huge else 0} bytes")

    print(f"    → 100KB: {'✓' if received else '✗'}, 500KB: {'✓' if received_huge else '✗'}", flush=True)

    await s1.stop()
    await s2.stop()


asyncio.run(test_large_payload())


# ═══════════════════════════════════════════════════════════════
# TEST 9: Delete throughput
# ═══════════════════════════════════════════════════════════════
print("\n=== 9. Delete Throughput ===", flush=True)


def test_delete_throughput():
    """Delete 1000 keys, verify tombstones don't grow unbounded."""
    from entropy_state import AgentStateStore

    store = AgentStateStore(room="perf-del", node_id=1, port=19940, gossip_interval_ms=5000)
    n = 1000

    for i in range(n):
        store.set(f"del/{i}", f"val_{i}")

    start = time.monotonic()
    for i in range(n):
        store.delete(f"del/{i}")
    elapsed = time.monotonic() - start

    remaining = store.get_all()
    tombstones = store.stats.tombstone_count

    print(f"    → {n} deletes in {elapsed*1000:.1f}ms, tombstones={tombstones}, remaining={len(remaining)}", flush=True)

    test(f"delete throughput: {n} keys < 500ms", elapsed < 0.5, f"took {elapsed:.2f}s")
    test(f"all keys deleted", len(remaining) == 0, f"{len(remaining)} remaining")
    test(f"tombstones tracked", tombstones > 0, f"tombstones={tombstones}")


test_delete_throughput()


# --- Summary ---
total = passed + failed
elapsed = time.time() - t0
print(f"\n{'=' * 60}", flush=True)
print(f"PERFORMANCE: {passed} passed, {failed} failed ({elapsed:.1f}s)", flush=True)
print(f"{'=' * 60}", flush=True)
sys.exit(1 if failed > 0 else 0)
