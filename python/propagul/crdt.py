"""OR-Map CRDT — Observed-Remove Map (Pure Python).

A conflict-free replicated key-value store with true deletion support.
Each key maps to a set of (value, unique-tag) pairs. Concurrent writes
to the same key both survive; a remove only affects tags the remover
has observed.

Wire-compatible with the Rust implementation (propagul-core).
Snapshot format: {"entries": {key: [{value, tag}]}, "tombstones": [[node_id, seq]]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Safety caps to prevent unbounded growth (identical to Rust)
MAX_TOMBSTONES = 50_000
MAX_ENTRIES_PER_KEY = 64
MAX_KEYS = 4_096

# Tag = (node_id, sequence_number)
Tag = Tuple[int, int]


@dataclass
class Entry:
    """A single value with its unique tag."""
    value: str
    tag: Tag


class ORMap:
    """OR-Map: Observed-Remove Map CRDT.

    Provides conflict-free replicated key-value semantics:
    - set(key, value): adds (value, unique-tag), removes locally-known tags for key
    - delete(key): removes all locally-known tags for key
    - merge(remote): union of entries minus tombstoned tags

    Merge is commutative, associative, and idempotent.
    """

    def __init__(self, node_id: int) -> None:
        self._node_id = node_id
        self._seq = 0
        self._entries: Dict[str, List[Entry]] = {}
        self._tombstones: Set[Tag] = set()

    # ─── Write Operations ────────────────────────────────────────

    def set(self, key: str, value: str) -> None:
        """Set a key to a value (observed-remove semantics)."""
        if key not in self._entries and len(self._entries) >= MAX_KEYS:
            return  # Silently drop to enforce bounded memory

        self._seq += 1
        tag: Tag = (self._node_id, self._seq)

        # Remove all existing entries for this key (add to tombstones)
        if key in self._entries:
            for entry in self._entries[key]:
                self._tombstones.add(entry.tag)

        # Add new entry
        self._entries[key] = [Entry(value=value, tag=tag)]
        self._gc_tombstones()

    def delete(self, key: str) -> None:
        """Delete a key (observed-remove: only removes tags we've seen)."""
        if key in self._entries:
            for entry in self._entries.pop(key):
                self._tombstones.add(entry.tag)
        self._gc_tombstones()

    # ─── Read Operations ─────────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        """Get current value. If conflicts exist, returns deterministic winner (highest tag)."""
        entries = self._entries.get(key)
        if not entries:
            return None
        # Deterministic: highest (node_id, seq) wins
        best = max(entries, key=lambda e: e.tag)
        return best.value

    def get_all(self, key: str) -> List[str]:
        """Get all concurrent values for a key (conflict detection)."""
        entries = self._entries.get(key, [])
        return [e.value for e in entries]

    def keys(self) -> List[str]:
        """All active keys."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def state_total(self) -> int:
        """Monotonic counter for state size tracking."""
        return self._seq

    @property
    def tombstone_count(self) -> int:
        return len(self._tombstones)

    # ─── Merge (CRDT Core) ───────────────────────────────────────

    def merge(self, remote_snapshot: dict) -> int:
        """Merge a remote snapshot. Returns delta (new entries added).

        CRDT invariant: merge is commutative, associative, idempotent
        within the limits of MAX_TOMBSTONES / MAX_KEYS / MAX_ENTRIES_PER_KEY.
        Note: caps make merge order-dependent under saturation — this is
        a documented trade-off for bounded memory.
        """
        if not isinstance(remote_snapshot, dict):
            return 0

        remote_entries = remote_snapshot.get("entries")
        if not isinstance(remote_entries, dict):
            remote_entries = {}

        remote_tombstones_raw = remote_snapshot.get("tombstones")
        if not isinstance(remote_tombstones_raw, list):
            remote_tombstones_raw = []

        delta = 0

        # 1. Add remote entries not in our tombstones (with validation)
        for key, r_entries in remote_entries.items():
            if not isinstance(key, str) or not isinstance(r_entries, list):
                continue

            if key not in self._entries and len(self._entries) >= MAX_KEYS:
                continue  # F-10: silently drop — delta stays accurate

            local = self._entries.setdefault(key, [])
            local_tags = {e.tag for e in local}  # O(1) lookup instead of O(n)

            for r_entry_raw in r_entries:
                if not isinstance(r_entry_raw, dict):
                    continue
                r_tag_raw = r_entry_raw.get("tag")
                r_value = r_entry_raw.get("value")

                # Validate tag: must be [int, int]
                if (
                    not isinstance(r_tag_raw, (list, tuple))
                    or len(r_tag_raw) != 2
                    or not isinstance(r_tag_raw[0], int)
                    or not isinstance(r_tag_raw[1], int)
                ):
                    continue
                # Validate value: must be string
                if not isinstance(r_value, str):
                    continue

                r_tag = (r_tag_raw[0], r_tag_raw[1])

                if r_tag not in self._tombstones and r_tag not in local_tags:
                    if len(local) < MAX_ENTRIES_PER_KEY:
                        local.append(Entry(value=r_value, tag=r_tag))
                        local_tags.add(r_tag)
                        delta += 1

        # 2. Absorb remote tombstones (capped during iteration)
        for t in remote_tombstones_raw:
            if len(self._tombstones) >= MAX_TOMBSTONES:
                break
            # Validate: must be [int, int]
            if (
                not isinstance(t, (list, tuple))
                or len(t) != 2
                or not isinstance(t[0], int)
                or not isinstance(t[1], int)
            ):
                continue
            self._tombstones.add((t[0], t[1]))

        # 3. Apply ALL tombstones to entries
        for key in list(self._entries.keys()):
            self._entries[key] = [
                e for e in self._entries[key]
                if e.tag not in self._tombstones
            ]

        # 4. Remove empty keys
        self._entries = {k: v for k, v in self._entries.items() if v}

        self._gc_tombstones()
        return delta

    # ─── Snapshot (Wire-Compatible) ──────────────────────────────

    def snapshot(self) -> dict:
        """Create a snapshot dict (wire-compatible with Rust implementation).

        Format: {"entries": {key: [{"value": str, "tag": [node_id, seq]}]},
                 "tombstones": [[node_id, seq], ...]}
        """
        return {
            "entries": {
                key: [{"value": e.value, "tag": list(e.tag)} for e in entries]
                for key, entries in self._entries.items()
            },
            "tombstones": [list(t) for t in self._tombstones],
        }

    def snapshot_bytes(self) -> bytes:
        """Serialize snapshot to bytes (JSON, wire-compatible)."""
        return json.dumps(self.snapshot(), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def parse_snapshot(data: bytes) -> dict:
        """Parse snapshot bytes into dict safely."""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}

    # ─── Tombstone GC ────────────────────────────────────────────

    def _gc_tombstones(self) -> None:
        """Evict oldest tombstones when cap exceeded."""
        if len(self._tombstones) <= MAX_TOMBSTONES:
            return
        # Sort by seq first (t[1]), then node_id (t[0]) for global determinism
        sorted_tags = sorted(self._tombstones, key=lambda t: (t[1], t[0]))
        to_remove = len(sorted_tags) - MAX_TOMBSTONES
        for tag in sorted_tags[:to_remove]:
            self._tombstones.discard(tag)

    # ─── Repr ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ORMap(node={self._node_id}, keys={len(self)}, seq={self._seq})"
