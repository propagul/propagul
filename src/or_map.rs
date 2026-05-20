/// OR-Map CRDT — Observed-Remove Map
///
/// A conflict-free replicated key-value store with true deletion support.
/// Each key maps to a set of (value, unique-tag) pairs. Concurrent writes
/// to the same key both survive; a remove only affects tags the remover
/// has observed.
///
/// Semantics:
/// - set(key, value): adds a new (value, tag) and removes all locally-known tags for key
/// - delete(key): removes all locally-known tags for key
/// - merge(remote): union of all entries, minus entries in the remote's tombstone set
///
/// This is strictly more powerful than LWW-Map: it handles concurrent
/// edits correctly without relying on synchronized clocks.

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

// Safety caps to prevent unbounded growth
const MAX_TOMBSTONES: usize = 50_000;
const MAX_ENTRIES_PER_KEY: usize = 64;
const MAX_KEYS: usize = 4_096;

/// Unique tag = (node_id, sequence_number)
pub type Tag = (u64, u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub value: String,
    pub tag: Tag,
}

/// OR-Map: Observed-Remove Map CRDT
#[derive(Clone, Debug)]
pub struct OrMap {
    node_id: u64,
    seq: u64,
    /// Active entries: key → set of (value, tag)
    entries: HashMap<String, Vec<Entry>>,
    /// Tombstones: tags that have been removed
    tombstones: HashSet<Tag>,
}

impl OrMap {
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            seq: 0,
            entries: HashMap::new(),
            tombstones: HashSet::new(),
        }
    }

    /// Set a key to a value. Removes all previous entries for this key
    /// (observed-remove semantics) and adds a new entry with a fresh tag.
    pub fn set(&mut self, key: &str, value: &str) {
        self.seq += 1;
        let tag: Tag = (self.node_id, self.seq);

        // Remove all existing entries for this key (add to tombstones)
        if let Some(existing) = self.entries.get(key) {
            for entry in existing {
                self.tombstones.insert(entry.tag);
            }
        }

        // Add new entry
        self.entries.insert(
            key.to_string(),
            vec![Entry {
                value: value.to_string(),
                tag,
            }],
        );

        self.gc_tombstones();
    }

    /// Delete a key (observed-remove: only removes tags we've seen).
    pub fn delete(&mut self, key: &str) {
        if let Some(existing) = self.entries.remove(key) {
            for entry in existing {
                self.tombstones.insert(entry.tag);
            }
        }
        self.gc_tombstones();
    }

    /// Get the current value for a key. If multiple concurrent values exist
    /// (conflict), returns the one with the highest tag (deterministic).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).and_then(|entries| {
            entries
                .iter()
                .max_by_key(|e| e.tag)
                .map(|e| e.value.as_str())
        })
    }

    /// Get all active entries for a key (includes conflicts).
    pub fn get_all(&self, key: &str) -> Vec<&str> {
        self.entries
            .get(key)
            .map(|entries| entries.iter().map(|e| e.value.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all keys currently in the map.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    /// Number of active keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute a total "state size" metric for EntropyAgent saturation tracking.
    pub fn state_total(&self) -> i64 {
        self.seq as i64
    }

    /// Create a serializable snapshot for gossip transmission.
    pub fn snapshot(&self) -> OrMapSnapshot {
        OrMapSnapshot {
            entries: self.entries.clone(),
            tombstones: self.tombstones.clone(),
        }
    }

    /// Merge a remote snapshot into this map. Returns the number of new entries added.
    ///
    /// CRDT invariant: merge is commutative, associative, idempotent.
    /// Tombstones are applied per-tag (not per-key) to preserve concurrent adds.
    pub fn merge(&mut self, remote: &OrMapSnapshot) -> i64 {
        let mut delta: i64 = 0;

        // 1. Add all remote entries that are not in our tombstones
        for (key, remote_entries) in &remote.entries {
            // Cap: reject new keys beyond MAX_KEYS
            if !self.entries.contains_key(key) && self.entries.len() >= MAX_KEYS {
                delta -= 1;  // Signal to caller: keys were dropped
                continue;
            }
            let local = self.entries.entry(key.clone()).or_default();
            for r_entry in remote_entries {
                if !self.tombstones.contains(&r_entry.tag) {
                    let already_exists = local.iter().any(|l| l.tag == r_entry.tag);
                    if !already_exists && local.len() < MAX_ENTRIES_PER_KEY {
                        local.push(r_entry.clone());
                        delta += 1;
                    }
                }
            }
        }

        // 2. Absorb remote tombstones (capped to prevent DoS)
        for tombstone in &remote.tombstones {
            if self.tombstones.len() >= MAX_TOMBSTONES {
                // Cap reached — skip remaining remote tombstones to prevent memory DoS
                break;
            }
            self.tombstones.insert(*tombstone);
        }

        // 3. Apply ALL tombstones (local + remote) to entries — single pass
        for entries in self.entries.values_mut() {
            entries.retain(|e| !self.tombstones.contains(&e.tag));
        }

        // 4. Remove empty keys
        self.entries.retain(|_, v| !v.is_empty());

        self.gc_tombstones();
        delta
    }

    /// Garbage-collect tombstones when the set exceeds MAX_TOMBSTONES.
    /// Evicts oldest tags (lowest seq numbers) to cap memory growth.
    ///
    /// Trade-off: evicting old tombstones means very old deletes can
    /// be "resurrected" if a stale peer merges them after GC. This is
    /// acceptable because:
    /// 1. Agent state is short-lived (task durations, not years)
    /// 2. 10k tombstones covers ~5000 set+delete cycles — sufficient for any session
    /// 3. Infrastructure Peers (Phase 4) provide authoritative state
    fn gc_tombstones(&mut self) {
        if self.tombstones.len() <= MAX_TOMBSTONES {
            return;
        }
        // Evict lowest-seq tombstones first (oldest operations)
        let mut sorted: Vec<Tag> = self.tombstones.iter().copied().collect();
        sorted.sort_by_key(|t| t.1); // sort by seq ascending
        let to_remove = sorted.len() - MAX_TOMBSTONES;
        for tag in sorted.iter().take(to_remove) {
            self.tombstones.remove(tag);
        }
    }

    /// Number of tombstones (for monitoring/debugging).
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }
}

/// Serializable snapshot for network transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrMapSnapshot {
    pub entries: HashMap<String, Vec<Entry>>,
    pub tombstones: HashSet<Tag>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut map = OrMap::new(1);
        map.set("name", "Alice");
        assert_eq!(map.get("name"), Some("Alice"));
    }

    #[test]
    fn test_overwrite() {
        let mut map = OrMap::new(1);
        map.set("name", "Alice");
        map.set("name", "Bob");
        assert_eq!(map.get("name"), Some("Bob"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_delete() {
        let mut map = OrMap::new(1);
        map.set("name", "Alice");
        map.delete("name");
        assert_eq!(map.get("name"), None);
        assert!(map.is_empty());
    }

    #[test]
    fn test_merge_adds_remote_entries() {
        let mut a = OrMap::new(1);
        let mut b = OrMap::new(2);
        a.set("x", "from-a");
        b.set("y", "from-b");

        let snap_b = b.snapshot();
        let delta = a.merge(&snap_b);

        assert!(delta > 0);
        assert_eq!(a.get("x"), Some("from-a"));
        assert_eq!(a.get("y"), Some("from-b"));
    }

    #[test]
    fn test_concurrent_writes_both_survive() {
        let mut a = OrMap::new(1);
        let mut b = OrMap::new(2);
        a.set("color", "red");
        b.set("color", "blue");

        let snap_b = b.snapshot();
        a.merge(&snap_b);

        // Both values exist as conflict
        let all = a.get_all("color");
        assert_eq!(all.len(), 2);
        assert!(all.contains(&"red"));
        assert!(all.contains(&"blue"));

        // get() returns deterministic winner (highest tag)
        assert!(a.get("color").is_some());
    }

    #[test]
    fn test_delete_propagates_via_tombstone() {
        let mut a = OrMap::new(1);
        let mut b = OrMap::new(2);
        a.set("temp", "data");

        // B receives A's data
        let snap_a = a.snapshot();
        b.merge(&snap_a);
        assert_eq!(b.get("temp"), Some("data"));

        // A deletes
        a.delete("temp");

        // B receives A's tombstone
        let snap_a2 = a.snapshot();
        b.merge(&snap_a2);
        assert_eq!(b.get("temp"), None);
    }

    #[test]
    fn test_merge_idempotent() {
        let mut a = OrMap::new(1);
        let mut b = OrMap::new(2);
        b.set("key", "val");

        let snap = b.snapshot();
        a.merge(&snap);
        let delta2 = a.merge(&snap);

        assert_eq!(delta2, 0, "Second merge of same snapshot should be idempotent");
    }

    #[test]
    fn test_state_total_monotonic() {
        let mut map = OrMap::new(1);
        let t0 = map.state_total();
        map.set("a", "1");
        let t1 = map.state_total();
        map.set("b", "2");
        let t2 = map.state_total();
        assert!(t0 < t1);
        assert!(t1 < t2);
    }
}
