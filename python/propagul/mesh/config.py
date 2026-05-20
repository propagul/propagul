"""propagul.mesh.config — CRDT-backed fleet configuration state.

Wraps an ORMap CRDT to provide fleet-wide configuration that survives
network partitions, offline nodes, concurrent edits, and server restarts.

Use-cases:
    - Desired model state: "llama3 should be on all nodes"
    - Node preferences: "node-a prefers GPU 0 for inference"
    - Fleet settings: "default context length = 4096"

Keys use namespace prefixes for isolation:
    desired:model:{name}    → "pull" or "delete"
    node:{id}:pref:{key}    → preference value
    fleet:setting:{key}     → fleet-wide setting

Persistence:
    CRDT snapshot is saved to disk as JSON on every mutation.
    On startup, the snapshot is loaded and merged back into the CRDT.
    Atomic writes (tmp + os.replace) prevent corruption.

Config state is synchronized via HTTP push/pull in the heartbeat cycle
(Phase 3.6). Full TCP gossip transport is planned for Phase 4.0.

Note: This module was moved from propagul.server.config_sync to
propagul.mesh.config (v0.13.26) to fix a packaging dependency:
the agent (mesh package, shipped via PyPI) had a hard import on
server code (excluded from PyPI). The class itself only depends on
propagul.crdt.ORMap (SDK code) and stdlib.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

from propagul.crdt import ORMap

logger = logging.getLogger("propagul.mesh.config")

# Namespace prefixes
_NS_DESIRED_MODEL = "desired:model:"
_NS_NODE_PREF = "node:{node_id}:pref:"
_NS_FLEET_SETTING = "fleet:setting:"


class FleetConfigMap:
    """CRDT-backed fleet configuration state.

    Provides eventually-consistent, conflict-free configuration
    that survives network partitions and concurrent edits.

    This is the core differentiator over commodity HTTP dashboards:
    config changes propagate to all nodes without a central coordinator,
    and offline nodes catch up when they reconnect.

    Args:
        node_id: Unique numeric ID for CRDT tag generation.
            Server uses 0, node agents use their hashed node_id.
    """

    def __init__(
        self,
        node_id: int = 0,
        persist_path: Optional[str] = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        self._crdt = ORMap(node_id)
        self._persist_path = persist_path
        self._debounce_seconds = debounce_seconds
        self._last_save_time: float = 0.0  # monotonic
        self._dirty = False  # Tracks unsaved mutations

        # Load persisted state on init
        if persist_path:
            self.load_from_disk()

    # ─── Desired Model State ──────────────────────────────────────

    def set_desired_model(self, model: str, action: str = "pull") -> None:
        """Set desired state for a model across the fleet.

        Args:
            model: Model name (e.g., "llama3:8b").
            action: "pull" (ensure present) or "delete" (ensure removed).

        Raises:
            ValueError: If action is not 'pull' or 'delete'.
        """
        if action not in ("pull", "delete"):
            raise ValueError(f"action must be 'pull' or 'delete', got '{action}'")
        key = f"{_NS_DESIRED_MODEL}{model}"
        self._crdt.set(key, action)
        self._auto_save()
        logger.info("Desired model set: %s → %s", model, action)

    def remove_desired_model(self, model: str) -> None:
        """Remove a model from desired state (no longer managed)."""
        key = f"{_NS_DESIRED_MODEL}{model}"
        self._crdt.delete(key)
        self._auto_save()

    def get_desired_models(self) -> Dict[str, str]:
        """Get all desired model states.

        Returns:
            dict mapping model name → action ("pull" or "delete").
        """
        result = {}
        for key in self._crdt.keys():
            if key.startswith(_NS_DESIRED_MODEL):
                model_name = key[len(_NS_DESIRED_MODEL):]
                val = self._crdt.get(key)
                if val is not None:
                    result[model_name] = val
        return result

    # ─── Node Preferences ─────────────────────────────────────────

    def set_node_preference(self, node_id: str, pref_key: str, value: str) -> None:
        """Set a preference for a specific node.

        Args:
            node_id: The node's string ID (e.g., "gpu-box-01").
            pref_key: Preference key (e.g., "gpu_affinity", "max_concurrent").
            value: Preference value as string.
        """
        key = f"node:{node_id}:pref:{pref_key}"
        self._crdt.set(key, value)
        self._auto_save()

    def get_node_preferences(self, node_id: str) -> Dict[str, str]:
        """Get all preferences for a node.

        Returns:
            dict mapping pref_key → value.
        """
        prefix = f"node:{node_id}:pref:"
        result = {}
        for key in self._crdt.keys():
            if key.startswith(prefix):
                pref_key = key[len(prefix):]
                val = self._crdt.get(key)
                if val is not None:
                    result[pref_key] = val
        return result

    # ─── Fleet Settings ───────────────────────────────────────────

    def set_fleet_setting(self, setting_key: str, value: str) -> None:
        """Set a fleet-wide configuration value.

        Args:
            setting_key: Setting key (e.g., "default_context_length").
            value: Setting value as string.
        """
        key = f"{_NS_FLEET_SETTING}{setting_key}"
        self._crdt.set(key, value)
        self._auto_save()

    def get_fleet_setting(self, setting_key: str) -> Optional[str]:
        """Get a fleet-wide configuration value."""
        key = f"{_NS_FLEET_SETTING}{setting_key}"
        return self._crdt.get(key)

    def get_all_fleet_settings(self) -> Dict[str, str]:
        """Get all fleet-wide settings."""
        result = {}
        for key in self._crdt.keys():
            if key.startswith(_NS_FLEET_SETTING):
                setting_key = key[len(_NS_FLEET_SETTING):]
                val = self._crdt.get(key)
                if val is not None:
                    result[setting_key] = val
        return result

    # ─── CRDT Operations ─────────────────────────────────────────

    def snapshot(self) -> dict:
        """Get CRDT snapshot for sync (wire-compatible)."""
        return self._crdt.snapshot()

    def merge(self, remote_snapshot: dict) -> int:
        """Merge a remote config snapshot.

        Returns delta (number of new entries).
        CRDT guarantees: commutative, associative, idempotent.
        Persists after merge if delta > 0.
        """
        delta = self._crdt.merge(remote_snapshot)
        if delta > 0:
            self._auto_save()
        return delta

    def key_count(self) -> int:
        """Number of active config entries."""
        return len(self._crdt)

    # ─── Summary ──────────────────────────────────────────────────

    def summary(self) -> dict:
        """Human-readable summary of config state."""
        desired = self.get_desired_models()
        return {
            "total_keys": len(self._crdt),
            "desired_models": len(desired),
            "desired_pull": sum(1 for v in desired.values() if v == "pull"),
            "desired_delete": sum(1 for v in desired.values() if v == "delete"),
            "tombstone_count": self._crdt.tombstone_count,
        }

    def __repr__(self) -> str:
        return f"FleetConfigMap(keys={len(self._crdt)}, tombstones={self._crdt.tombstone_count})"

    # ─── Persistence ────────────────────────────────────────────

    def _auto_save(self) -> None:
        """Save after mutations if persist_path is set.

        Uses time-based debouncing to coalesce rapid writes.
        At most one disk write per debounce_seconds (default: 2s).

        The dirty flag ensures no mutation is lost: if a save is
        skipped due to debouncing, the next mutation (or explicit
        flush_to_disk() call) will capture the accumulated state.

        Worst case: debounce_seconds of data at risk if process
        crashes between mutations. For config data (not telemetry),
        this is acceptable — config changes happen at human speed.
        """
        if not self._persist_path:
            return

        self._dirty = True
        now = time.monotonic()
        elapsed = now - self._last_save_time

        if elapsed >= self._debounce_seconds:
            self.save_to_disk()

    def save_to_disk(self) -> bool:
        """Save CRDT snapshot to disk.

        Uses atomic write (write to tmp, then os.replace) to prevent
        corruption during crashes. Returns True on success.
        """
        if not self._persist_path:
            return False

        snapshot = self._crdt.snapshot()
        data = {
            "version": 1,
            "snapshot": snapshot,
        }

        tmp_path = self._persist_path + ".tmp"
        try:
            parent = os.path.dirname(self._persist_path) or "."
            os.makedirs(parent, exist_ok=True)

            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._persist_path)
            self._last_save_time = time.monotonic()
            self._dirty = False
            return True
        except Exception as e:
            logger.error("Failed to save config map: %s", e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False

    def load_from_disk(self) -> int:
        """Load CRDT snapshot from disk and merge into current state.

        Uses merge (not replace) so that any in-memory state from
        concurrent operations is preserved. Returns delta count.
        """
        if not self._persist_path or not os.path.isfile(self._persist_path):
            return 0

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            if data.get("version") != 1:
                logger.warning("Unknown config map version: %s", data.get("version"))
                return 0

            snapshot = data.get("snapshot")
            if not isinstance(snapshot, dict):
                logger.warning("Invalid config map snapshot")
                return 0

            delta = self._crdt.merge(snapshot)
            logger.info(
                "Config map loaded from disk: %d keys, %d delta",
                len(self._crdt), delta,
            )
            return delta

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Corrupt config map file: %s", e)
            return 0
        except Exception as e:
            logger.error("Failed to load config map: %s", e)
            return 0

    def flush_to_disk(self) -> bool:
        """Force-save if any unsaved mutations exist.

        Call this on graceful shutdown to ensure the last
        debounced mutations are persisted.
        """
        if self._dirty:
            return self.save_to_disk()
        return False

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved mutations (debounce pending)."""
        return self._dirty
