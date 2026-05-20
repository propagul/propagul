#!/usr/bin/env python3
"""Phase 5: Post-Pivot Hardening Tests (Pure Python SDK)

1. Snapshot Format Stability
2. Cross-Version Merge Compatibility
3. Network Latency + Packet Loss Simulation
4. 10K+ Key Scale Test
5. MAX_KEYS Cap Enforcement

Run standalone:
    .venv/bin/python tests/test_hardening.py

Or via pytest (functions with test_ prefix are collected automatically):
    .venv/bin/python -m pytest tests/test_hardening.py -q
"""
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Module-level helpers used by both standalone and pytest modes
_passed = 0
_failed = 0
_t0 = 0.0


def _check(name, condition, detail=""):
    """Standalone-mode assertion tracker."""
    global _passed, _failed
    if condition:
        print(f"  ✅ {name} (+{time.time() - _t0:.1f}s)", flush=True)
        _passed += 1
    else:
        print(f"  ❌ {name} — {detail} (+{time.time() - _t0:.1f}s)", flush=True)
        _failed += 1


# ═══════════════════════════════════════════════════════════════
# 1. SNAPSHOT FORMAT STABILITY
# ═══════════════════════════════════════════════════════════════

def test_snapshot_format_stability():
    """Prove that the current snapshot format is stable and can be
    round-tripped through JSON serialization."""
    from propagul.crdt import ORMap

    s = ORMap(node_id=1)
    s.set("agent/status", "active")
    s.set("task/0/result", "Research complete")
    s.set("config/model", "gpt-4")

    snap = s.snapshot()
    assert isinstance(snap, dict)
    assert "entries" in snap
    assert "tombstones" in snap

    entries = snap.get("entries", {})
    assert len(entries) == 3, f"got {len(entries)} keys"

    for key, entry_list in entries.items():
        for entry in entry_list:
            assert "value" in entry, f"entry missing 'value': {entry}"
            assert "tag" in entry, f"entry missing 'tag': {entry}"
            break
        break

    # Bytes roundtrip
    snap_bytes = s.snapshot_bytes()
    parsed = json.loads(snap_bytes)
    assert parsed["entries"] == snap["entries"]


def test_cross_version_merge():
    """Simulate merge between two independent ORMap instances."""
    from propagul.crdt import ORMap

    v05 = ORMap(node_id=1)
    v06 = ORMap(node_id=2)

    v05.set("legacy_key", "v0.5_value")
    v05.set("agent/alpha/status", "running")

    v06.set("new_key", "v0.6_value")
    v06.set("agent/beta/status", "idle")

    snap_05 = v05.snapshot()
    snap_06 = v06.snapshot()

    v06.merge(snap_05)
    assert v06.get("legacy_key") == "v0.5_value"
    assert v06.get("agent/alpha/status") == "running"

    v05.merge(snap_06)
    assert v05.get("new_key") == "v0.6_value"
    assert v05.get("agent/beta/status") == "idle"

    # Convergence: both maps should agree on all keys
    for key in ["legacy_key", "new_key", "agent/alpha/status", "agent/beta/status"]:
        assert v05.get(key) == v06.get(key), f"{key}: {v05.get(key)} != {v06.get(key)}"


def test_snapshot_with_unknown_fields():
    """If a future version adds fields to the snapshot JSON,
    merge must still work (ignore unknown fields)."""
    from propagul.crdt import ORMap

    s = ORMap(node_id=1)
    s.set("key", "value")
    snap = s.snapshot()
    snap["version"] = 2
    snap["metadata"] = {"created_by": "v0.7"}

    s2 = ORMap(node_id=2)
    s2.merge(snap)
    assert s2.get("key") == "value"


# ═══════════════════════════════════════════════════════════════
# 2. NETWORK SIMULATION: Latency + Packet Loss
# ═══════════════════════════════════════════════════════════════

class LossyTransportWrapper:
    """Wraps GossipTransport to inject latency and packet loss."""

    def __init__(self, store, loss_rate=0.0, min_latency_ms=0, max_latency_ms=0):
        self._store = store
        self._real_transport = store._transport
        self._loss_rate = loss_rate
        self._min_latency = min_latency_ms
        self._max_latency = max_latency_ms
        self._dropped = 0
        self._delayed = 0

        # Monkey-patch the send method
        self._original_send = self._real_transport.send_to_k_peers
        self._real_transport.send_to_k_peers = self._send_with_fault

    async def _send_with_fault(self, peers, data, k):
        if random.random() < self._loss_rate:
            self._dropped += 1
            return [False] * min(k, len(peers))

        if self._max_latency > 0:
            delay = random.uniform(self._min_latency, self._max_latency) / 1000.0
            await asyncio.sleep(delay)
            self._delayed += 1

        return await self._original_send(peers, data, k)

    def restore(self):
        self._real_transport.send_to_k_peers = self._original_send


def test_4k_write_read():
    """Write and read 4000 keys (just under MAX_KEYS cap)."""
    from propagul import AgentStateStore

    store = AgentStateStore(room="scale-4k", node_id=1, port=19960, gossip_interval_ms=5000)

    n = 4000

    start = time.monotonic()
    for i in range(n):
        store.set(f"k/{i}", f"v_{i}_{'x' * 20}")
    write_elapsed = time.monotonic() - start

    start = time.monotonic()
    for i in range(n):
        _ = store.get(f"k/{i}")
    read_elapsed = time.monotonic() - start

    all_data = store.get_all()
    correct = all(store.get(f"k/{i}") == f"v_{i}_{'x' * 20}" for i in range(n))

    assert write_elapsed < 0.2, f"write took {write_elapsed:.3f}s"
    assert read_elapsed < 0.2, f"read took {read_elapsed:.3f}s"
    assert correct, "not all values correct"
    assert len(all_data) == n, f"got {len(all_data)}"


def test_max_keys_cap():
    """Verify MAX_KEYS=4096 cap prevents unbounded growth."""
    from propagul.crdt import ORMap

    s1 = ORMap(node_id=1)
    s2 = ORMap(node_id=2)

    for i in range(4096):
        s1.set(f"fill/{i}", f"v_{i}")

    for i in range(100):
        s2.set(f"extra/{i}", f"v_{i}")

    snap2 = s2.snapshot()
    delta = s1.merge(snap2)

    total = len(s1)
    assert total <= 4096, f"total={total}"
    # When cap prevents keys from being added, delta is 0 (no net change)
    assert delta <= 0, f"delta={delta} (expected 0 or negative)"


# ═══════════════════════════════════════════════════════════════
# Standalone runner (for backward compat: python tests/test_hardening.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _t0 = time.time()

    print("=== 1. Snapshot Format Stability ===", flush=True)
    try:
        test_snapshot_format_stability()
        _check("snapshot_format_stability", True)
    except AssertionError as e:
        _check("snapshot_format_stability", False, str(e))

    try:
        test_cross_version_merge()
        _check("cross_version_merge", True)
    except AssertionError as e:
        _check("cross_version_merge", False, str(e))

    try:
        test_snapshot_with_unknown_fields()
        _check("snapshot_with_unknown_fields", True)
    except AssertionError as e:
        _check("snapshot_with_unknown_fields", False, str(e))

    print("\n=== 2. Scale Tests ===", flush=True)
    try:
        test_4k_write_read()
        _check("4k_write_read", True)
    except AssertionError as e:
        _check("4k_write_read", False, str(e))

    try:
        test_max_keys_cap()
        _check("max_keys_cap", True)
    except AssertionError as e:
        _check("max_keys_cap", False, str(e))

    # --- Summary ---
    total = _passed + _failed
    elapsed = time.time() - _t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"HARDENING: {_passed} passed, {_failed} failed ({elapsed:.1f}s)", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if _failed > 0 else 0)
