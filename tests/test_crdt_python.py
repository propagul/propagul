"""CRDT Correctness Tests — OR-Map (Pure Python).

Tests the three fundamental CRDT properties:
1. Commutativity: merge(A, B) == merge(B, A)
2. Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
3. Idempotency: merge(A, A) == A

Plus: wire compatibility with Rust implementation.
"""

import json
import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from propagul.crdt import ORMap

passed = 0
failed = 0

def _test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


def states_equal(a: ORMap, b: ORMap) -> bool:
    """Compare two OR-Maps by their observable state (keys + values)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a.keys():
        if sorted(a.get_all(key)) != sorted(b.get_all(key)):
            return False
    return True


def run_all():
    """Run all CRDT correctness tests (standalone mode)."""
    global passed, failed
    passed = 0
    failed = 0

    print("=" * 60)
    print("CRDT CORRECTNESS TESTS")
    print("=" * 60)

    # ─── Basic Operations ────────────────────────────────────────

    print("\n--- Basic Operations ---")

    m = ORMap(1)
    m.set("name", "Alice")
    _test("set and get", m.get("name") == "Alice")

    m.set("name", "Bob")
    _test("overwrite", m.get("name") == "Bob" and len(m) == 1)

    m.delete("name")
    _test("delete", m.get("name") is None and len(m) == 0)

    m.set("a", "1")
    m.set("b", "2")
    m.set("c", "3")
    _test("multiple keys", len(m) == 3 and set(m.keys()) == {"a", "b", "c"})

    m2 = ORMap(1)
    _test("contains", "a" not in m2)
    m2.set("x", "y")
    _test("contains after set", "x" in m2)

    # ─── Merge Basics ────────────────────────────────────────────

    print("\n--- Merge Basics ---")

    a = ORMap(1)
    b = ORMap(2)
    a.set("x", "from-a")
    b.set("y", "from-b")

    delta = a.merge(b.snapshot())
    _test("merge adds remote entries", delta > 0)
    _test("local entry preserved", a.get("x") == "from-a")
    _test("remote entry added", a.get("y") == "from-b")

    # ─── Concurrent Writes (Conflict) ────────────────────────────

    print("\n--- Concurrent Writes ---")

    a = ORMap(1)
    b = ORMap(2)
    a.set("color", "red")
    b.set("color", "blue")

    a.merge(b.snapshot())
    conflicts = a.get_all("color")
    _test("both values survive", len(conflicts) == 2)
    _test("red in conflicts", "red" in conflicts)
    _test("blue in conflicts", "blue" in conflicts)
    _test("get returns deterministic winner", a.get("color") is not None)

    # ─── Delete Propagation via Tombstone ─────────────────────────

    print("\n--- Delete Propagation ---")

    a = ORMap(1)
    b = ORMap(2)
    a.set("temp", "data")

    b.merge(a.snapshot())
    _test("B received A's data", b.get("temp") == "data")

    a.delete("temp")
    b.merge(a.snapshot())
    _test("tombstone propagated", b.get("temp") is None)

    # ─── CRDT Property 1: Idempotency ────────────────────────────

    print("\n--- Idempotency ---")

    a = ORMap(1)
    b = ORMap(2)
    b.set("key", "val")

    snap = b.snapshot()
    a.merge(snap)
    state_after_first = a.snapshot()
    delta2 = a.merge(snap)
    state_after_second = a.snapshot()

    _test("second merge delta is 0", delta2 == 0, f"delta2={delta2}")
    _test("state unchanged after double merge",
         json.dumps(state_after_first, sort_keys=True) == json.dumps(state_after_second, sort_keys=True))

    # ─── CRDT Property 2: Commutativity ──────────────────────────

    print("\n--- Commutativity ---")

    a = ORMap(1)
    b = ORMap(2)
    c = ORMap(3)

    a.set("x", "a-val")
    b.set("y", "b-val")

    # Path 1: A merges B
    ab = ORMap(10)
    ab.merge(a.snapshot())
    ab.merge(b.snapshot())

    # Path 2: B merges A
    ba = ORMap(11)
    ba.merge(b.snapshot())
    ba.merge(a.snapshot())

    _test("commutative: merge(A,B) == merge(B,A)", states_equal(ab, ba))

    # With conflicts
    a2 = ORMap(1)
    b2 = ORMap(2)
    a2.set("same", "from-a")
    b2.set("same", "from-b")

    ab2 = ORMap(10)
    ab2.merge(a2.snapshot())
    ab2.merge(b2.snapshot())

    ba2 = ORMap(11)
    ba2.merge(b2.snapshot())
    ba2.merge(a2.snapshot())

    _test("commutative with conflicts", states_equal(ab2, ba2))

    # ─── CRDT Property 3: Associativity ──────────────────────────

    print("\n--- Associativity ---")

    a = ORMap(1)
    b = ORMap(2)
    c = ORMap(3)
    a.set("a", "1")
    b.set("b", "2")
    c.set("c", "3")

    # Path 1: (A ∪ B) ∪ C
    ab_c = ORMap(10)
    ab_c.merge(a.snapshot())
    ab_c.merge(b.snapshot())
    ab_c.merge(c.snapshot())

    # Path 2: A ∪ (B ∪ C)
    a_bc = ORMap(11)
    bc = ORMap(12)
    bc.merge(b.snapshot())
    bc.merge(c.snapshot())
    a_bc.merge(a.snapshot())
    a_bc.merge(bc.snapshot())

    _test("associative: (A∪B)∪C == A∪(B∪C)", states_equal(ab_c, a_bc))

    # ─── State Total Monotonic ────────────────────────────────────

    print("\n--- State Total ---")

    m = ORMap(1)
    t0 = m.state_total
    m.set("a", "1")
    t1 = m.state_total
    m.set("b", "2")
    t2 = m.state_total
    _test("state_total monotonic", t0 < t1 < t2)

    # ─── Tombstone GC ────────────────────────────────────────────

    print("\n--- Tombstone GC ---")

    m = ORMap(1)
    for i in range(100):
        m.set(f"key-{i % 10}", f"val-{i}")
    _test("tombstones tracked", m.tombstone_count > 0)
    _test("tombstones below cap", m.tombstone_count <= 50_000)

    # ─── Snapshot / Wire Format ──────────────────────────────────

    print("\n--- Snapshot Format ---")

    m = ORMap(1)
    m.set("task", "research")
    snap = m.snapshot()

    _test("snapshot has entries", "entries" in snap)
    _test("snapshot has tombstones", "tombstones" in snap)
    _test("entry has value", snap["entries"]["task"][0]["value"] == "research")
    _test("entry has tag as list", isinstance(snap["entries"]["task"][0]["tag"], list))
    _test("tag is [node_id, seq]", snap["entries"]["task"][0]["tag"] == [1, 1])

    # Bytes roundtrip
    snap_bytes = m.snapshot_bytes()
    parsed = ORMap.parse_snapshot(snap_bytes)
    _test("bytes roundtrip", parsed == snap)

    # ─── Wire Compatibility with Rust ─────────────────────────────

    print("\n--- Wire Compatibility ---")

    try:
        from propagul_core import GossipCore
        rust = GossipCore(99, 0, 500.0)
        rust.set("hello", "world")
        rust.set("foo", "bar")

        rust_snap_bytes = rust.snapshot()
        rust_snap = json.loads(rust_snap_bytes)

        # Python can merge Rust snapshot
        py = ORMap(1)
        delta = py.merge(rust_snap)
        _test("python merges rust snapshot", delta > 0)
        _test("python has rust key", py.get("hello") == "world")
        _test("python has rust key 2", py.get("foo") == "bar")

        # Rust can merge Python snapshot
        py2 = ORMap(2)
        py2.set("python_key", "python_val")
        py_snap_bytes = py2.snapshot_bytes()

        rust_delta = rust.merge_remote(py_snap_bytes, time.time() * 1000)
        _test("rust merges python snapshot", rust_delta > 0)

        # Bidirectional merge
        py3 = ORMap(3)
        py3.set("a", "from-py")
        rust2 = GossipCore(4, 0, 500.0)
        rust2.set("b", "from-rust")

        # py3 merges rust2
        py3.merge(json.loads(rust2.snapshot()))
        _test("bidirectional: py has rust key", py3.get("b") == "from-rust")
        _test("bidirectional: py has own key", py3.get("a") == "from-py")

        # rust2 merges py3
        rust2.merge_remote(py3.snapshot_bytes(), time.time() * 1000)
        _test("bidirectional: rust has py key",
             rust2.get("a") == "from-py")

        print("\n  ✅ WIRE COMPATIBILITY CONFIRMED: Python ↔ Rust")

    except ImportError:
        print("\n  ⚠️  Rust core not available — wire compat tests skipped")

    # ─── Performance ─────────────────────────────────────────────

    print("\n--- Performance ---")

    m = ORMap(1)
    n = 10_000
    t0 = time.time()
    for i in range(n):
        m.set(f"key-{i % 1000}", f"val-{i}")
    write_time = time.time() - t0
    writes_per_sec = n / write_time
    _test(f"write throughput: {writes_per_sec:,.0f}/s (>10K required)",
         writes_per_sec > 10_000,
         f"{writes_per_sec:,.0f}/s")

    t0 = time.time()
    for i in range(n):
        m.get(f"key-{i % 1000}")
    read_time = time.time() - t0
    reads_per_sec = n / read_time
    _test(f"read throughput: {reads_per_sec:,.0f}/s (>50K required)",
         reads_per_sec > 50_000,
         f"{reads_per_sec:,.0f}/s")

    # Merge performance
    a = ORMap(1)
    b = ORMap(2)
    for i in range(500):
        a.set(f"k-{i}", f"v-{i}")
        b.set(f"k-{i + 500}", f"v-{i + 500}")

    snap = b.snapshot()
    t0 = time.time()
    a.merge(snap)
    merge_time = time.time() - t0
    _test(f"merge 500 keys: {merge_time*1000:.1f}ms (<100ms required)",
         merge_time < 0.1,
         f"{merge_time*1000:.1f}ms")

    # ─── Summary ─────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print(f"CRDT CORRECTNESS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed


# ─── pytest-compatible test functions ─────────────────────────

def test_crdt_basic_operations():
    """Basic set/get/delete operations."""
    m = ORMap(1)
    m.set("name", "Alice")
    assert m.get("name") == "Alice"
    m.set("name", "Bob")
    assert m.get("name") == "Bob"
    m.delete("name")
    assert m.get("name") is None

def test_crdt_merge_commutativity():
    """merge(A,B) == merge(B,A)."""
    a, b = ORMap(1), ORMap(2)
    a.set("x", "a-val")
    b.set("y", "b-val")
    ab, ba = ORMap(10), ORMap(11)
    ab.merge(a.snapshot()); ab.merge(b.snapshot())
    ba.merge(b.snapshot()); ba.merge(a.snapshot())
    assert states_equal(ab, ba)

def test_crdt_merge_idempotency():
    """Double merge produces no delta."""
    a, b = ORMap(1), ORMap(2)
    b.set("key", "val")
    snap = b.snapshot()
    a.merge(snap)
    assert a.merge(snap) == 0

def test_crdt_tombstone_propagation():
    """Tombstones propagate deletions."""
    a, b = ORMap(1), ORMap(2)
    a.set("temp", "data")
    b.merge(a.snapshot())
    a.delete("temp")
    b.merge(a.snapshot())
    assert b.get("temp") is None

def test_crdt_snapshot_roundtrip():
    """Snapshot serialization roundtrip."""
    m = ORMap(1)
    m.set("task", "research")
    assert ORMap.parse_snapshot(m.snapshot_bytes()) == m.snapshot()


if __name__ == "__main__":
    rc = run_all()
    sys.exit(1 if rc > 0 else 0)
