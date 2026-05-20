"""propagul.integrations.langgraph — LangGraph Checkpointer backend.

Replaces LangGraph's default SQLite/Postgres checkpointer with
Propagul CRDT-based persistence. State is synchronized via gossip
instead of stored in a central database.

Usage:
    from propagul.integrations.langgraph import EntropyCheckpointer

    checkpointer = EntropyCheckpointer(room="my-graph", port=9001)
    graph = workflow.compile(checkpointer=checkpointer)
    result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})

Requires: pip install langgraph
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("propagul.langgraph")

# Lazy import guard
_langgraph_available = None


def _check_langgraph():
    global _langgraph_available
    if _langgraph_available is None:
        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401
            _langgraph_available = True
        except ImportError:
            _langgraph_available = False
    return _langgraph_available


def _get_base_class():
    """Get BaseCheckpointSaver, importing lazily."""
    from langgraph.checkpoint.base import BaseCheckpointSaver
    return BaseCheckpointSaver


def _get_types():
    """Import LangGraph checkpoint types lazily."""
    from langgraph.checkpoint.base import (
        CheckpointTuple,
        Checkpoint,
        CheckpointMetadata,
        ChannelVersions,
    )
    from langchain_core.runnables import RunnableConfig
    return CheckpointTuple, Checkpoint, CheckpointMetadata, ChannelVersions, RunnableConfig


# Build the class dynamically to avoid import errors when langgraph is not installed
def _build_checkpointer_class():
    """Build EntropyCheckpointer class inheriting from BaseCheckpointSaver.

    This avoids import-time failures when langgraph is not installed.
    """
    BaseCheckpointSaver = _get_base_class()
    CheckpointTuple, Checkpoint, CheckpointMetadata, ChannelVersions, RunnableConfig = _get_types()

    from langgraph.checkpoint.base.id import uuid6

    class _EntropyCheckpointer(BaseCheckpointSaver):
        """LangGraph-compatible checkpointer backed by Propagul CRDT.

        Inherits from BaseCheckpointSaver and implements the full interface
        required by LangGraph 1.2.0 / langgraph-checkpoint 4.1.0.

        State layout in CRDT:
        - "cp/{thread_id}/latest" → serialized checkpoint
        - "cp/{thread_id}/meta" → serialized metadata
        - "cp/{thread_id}/versions" → serialized new_versions
        - "cp/{thread_id}/parent" → parent checkpoint_id
        - "cp/{thread_id}/writes/{task_id}" → pending writes
        - "cp/{thread_id}/history/{slot}" → historical checkpoints (ring buffer)
        """

        def __init__(
            self,
            room: str = "langgraph",
            port: int = 9001,
            node_id: Optional[int] = None,
            peers: Optional[list] = None,
            max_history: int = 10,
            gossip_secret: Optional[bytes] = None,
        ):
            super().__init__()

            from propagul.store import AgentStateStore
            from propagul.types import PeerAddress

            self._node_id = node_id or (port * 31 + 7)
            self._max_history = max_history
            self._checkpoint_counter = 0

            self._store = AgentStateStore(
                room=room,
                node_id=self._node_id,
                port=port,
                gossip_interval_ms=200.0,
                gossip_secret=gossip_secret,
            )

            self._peers = []
            if peers:
                for p in peers:
                    if isinstance(p, PeerAddress):
                        self._peers.append(p)
                    elif isinstance(p, tuple):
                        self._peers.append(PeerAddress(p[0], p[1]))
                    elif isinstance(p, str) and ":" in p:
                        host, port_str = p.rsplit(":", 1)
                        self._peers.append(PeerAddress(host, int(port_str)))

            self._started = False

        def _ensure_started(self):
            """Lazy-start the gossip transport."""
            if not self._started:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    raise RuntimeError(
                        "EntropyCheckpointer._ensure_started() called from within "
                        "a running event loop. Use 'await checkpointer.astart()' "
                        "or start the store before entering the async context."
                    )
                except RuntimeError as e:
                    if "no running event loop" not in str(e).lower():
                        raise
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._store.start(peers=self._peers))
                    self._started = True
                    self._loop = loop

        async def astart(self):
            """Async start — use this when calling from an async context."""
            if not self._started:
                await self._store.start(peers=self._peers)
                self._started = True

        # ─── Core sync methods ────────────────────────────────────────

        def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            """Retrieve the latest checkpoint for a thread."""
            self._ensure_started()

            thread_id = config.get("configurable", {}).get("thread_id", "default")
            key_prefix = f"cp/{thread_id}"

            checkpoint_json = self._store.get(f"{key_prefix}/latest")
            if checkpoint_json is None:
                return None

            try:
                checkpoint = json.loads(checkpoint_json)
            except json.JSONDecodeError:
                logger.warning("Corrupt checkpoint for thread %s", thread_id)
                return None

            checkpoint_id = self._store.get(f"{key_prefix}/checkpoint_id") or ""
            meta_json = self._store.get(f"{key_prefix}/meta")
            metadata = json.loads(meta_json) if meta_json else {}

            parent_id = self._store.get(f"{key_prefix}/parent")
            parent_config = None
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_id,
                    }
                }

            # Load pending writes
            pending_writes = self._load_pending_writes(thread_id)

            result_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

            return CheckpointTuple(
                config=result_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

        def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
        ) -> Iterator[CheckpointTuple]:
            """List checkpoint history for a thread."""
            self._ensure_started()

            if config is None:
                return

            thread_id = config.get("configurable", {}).get("thread_id", "default")
            key_prefix = f"cp/{thread_id}"

            count = 0
            for i in range(self._max_history):
                if limit and count >= limit:
                    break

                history_json = self._store.get(f"{key_prefix}/history/{i}")
                if history_json is None:
                    continue

                try:
                    entry = json.loads(history_json)
                except json.JSONDecodeError:
                    continue

                entry_checkpoint_id = entry.get("checkpoint_id", "")

                # Apply 'before' filter
                if before:
                    before_id = before.get("configurable", {}).get("checkpoint_id")
                    if before_id and entry_checkpoint_id >= before_id:
                        continue

                entry_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": entry_checkpoint_id,
                    }
                }

                yield CheckpointTuple(
                    config=entry_config,
                    checkpoint=entry.get("checkpoint", {}),
                    metadata=entry.get("metadata", {}),
                )
                count += 1

        def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            """Store a checkpoint with its configuration and metadata."""
            self._ensure_started()

            thread_id = config.get("configurable", {}).get("thread_id", "default")

            # Generate checkpoint ID if not present
            if "id" not in checkpoint:
                checkpoint["id"] = str(uuid6())

            checkpoint_id = checkpoint["id"]
            key_prefix = f"cp/{thread_id}"

            # Store parent reference (current latest becomes parent)
            current_id = self._store.get(f"{key_prefix}/checkpoint_id")
            if current_id:
                self._store.set(f"{key_prefix}/parent", current_id)

            # Serialize and store
            self._store.set(f"{key_prefix}/latest", json.dumps(checkpoint, default=str))
            self._store.set(f"{key_prefix}/checkpoint_id", checkpoint_id)
            self._store.set(f"{key_prefix}/ts", str(time.time()))
            self._store.set(f"{key_prefix}/meta", json.dumps(metadata, default=str))
            self._store.set(f"{key_prefix}/versions", json.dumps(new_versions, default=str))

            # Store in history (ring buffer)
            self._checkpoint_counter += 1
            history_slot = self._checkpoint_counter % self._max_history
            self._store.set(
                f"{key_prefix}/history/{history_slot}",
                json.dumps({
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": checkpoint,
                    "metadata": metadata,
                    "ts": time.time(),
                }, default=str),
            )

            logger.debug("Stored checkpoint %s for thread %s", checkpoint_id, thread_id)

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

        def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            """Store pending writes for a checkpoint."""
            self._ensure_started()

            thread_id = config.get("configurable", {}).get("thread_id", "default")
            key = f"cp/{thread_id}/writes/{task_id}"

            serialized = json.dumps(
                [{"channel": ch, "value": val, "task_path": task_path} for ch, val in writes],
                default=str,
            )
            self._store.set(key, serialized)

        def delete_thread(self, thread_id: str) -> None:
            """Delete all checkpoints for a thread."""
            self._ensure_started()

            key_prefix = f"cp/{thread_id}"
            for suffix in ["/latest", "/checkpoint_id", "/ts", "/meta", "/versions", "/parent"]:
                self._store.delete(f"{key_prefix}{suffix}")
            for i in range(self._max_history):
                self._store.delete(f"{key_prefix}/history/{i}")

        def delete_for_runs(self, run_ids: Sequence[str]) -> None:
            """Delete checkpoints for specific runs (no-op: CRDT doesn't track run IDs)."""
            logger.debug("delete_for_runs called — no-op in CRDT backend")

        def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
            """Copy all checkpoints from one thread to another."""
            self._ensure_started()

            src_prefix = f"cp/{source_thread_id}"
            tgt_prefix = f"cp/{target_thread_id}"

            for suffix in ["/latest", "/checkpoint_id", "/ts", "/meta", "/versions", "/parent"]:
                val = self._store.get(f"{src_prefix}{suffix}")
                if val is not None:
                    self._store.set(f"{tgt_prefix}{suffix}", val)

            for i in range(self._max_history):
                val = self._store.get(f"{src_prefix}/history/{i}")
                if val is not None:
                    self._store.set(f"{tgt_prefix}/history/{i}", val)

        def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
            """Prune checkpoint history for given threads."""
            self._ensure_started()

            for thread_id in thread_ids:
                if strategy == "delete":
                    self.delete_thread(thread_id)
                elif strategy == "keep_latest":
                    key_prefix = f"cp/{thread_id}"
                    for i in range(self._max_history):
                        self._store.delete(f"{key_prefix}/history/{i}")

        # ─── Async variants (delegate to sync) ───────────────────────

        async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            return self.get_tuple(config)

        async def alist(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
        ) -> AsyncIterator[CheckpointTuple]:
            for item in self.list(config, filter=filter, before=before, limit=limit):
                yield item

        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            return self.put(config, checkpoint, metadata, new_versions)

        async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            self.put_writes(config, writes, task_id, task_path)

        async def adelete_thread(self, thread_id: str) -> None:
            self.delete_thread(thread_id)

        async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
            self.delete_for_runs(run_ids)

        async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
            self.copy_thread(source_thread_id, target_thread_id)

        async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
            self.prune(thread_ids, strategy=strategy)

        # ─── Helpers ──────────────────────────────────────────────────

        def _load_pending_writes(self, thread_id: str) -> list:
            """Load all pending writes for a thread."""
            pending = []
            all_keys = self._store.get_all()
            prefix = f"cp/{thread_id}/writes/"
            for key, val in all_keys.items():
                if key.startswith(prefix):
                    task_id = key[len(prefix):]
                    try:
                        writes = json.loads(val)
                        for w in writes:
                            pending.append((task_id, w.get("channel", ""), w.get("value")))
                    except json.JSONDecodeError:
                        pass
            return pending if pending else None

        @property
        def stats(self):
            """Get Propagul telemetry."""
            return self._store.stats

        def close(self) -> None:
            """Stop gossip transport (sync)."""
            if self._started and hasattr(self, '_loop'):
                try:
                    self._loop.run_until_complete(self._store.stop())
                    self._loop.close()
                except Exception:
                    pass
                self._started = False

        async def aclose(self) -> None:
            """Stop gossip transport (async)."""
            if self._started:
                await self._store.stop()
                self._started = False

        def __del__(self):
            if self._started and hasattr(self, '_loop'):
                self.close()

    return _EntropyCheckpointer


# Public API: lazy class construction
class EntropyCheckpointer:
    """LangGraph-compatible checkpointer backed by Propagul CRDT.

    This is a factory that returns a proper BaseCheckpointSaver subclass
    when langgraph is available, or raises ImportError when it's not.

    Usage:
        checkpointer = EntropyCheckpointer(room="my-graph", port=9001)
        graph = workflow.compile(checkpointer=checkpointer)
    """

    _real_class = None

    def __new__(cls, *args, **kwargs):
        if not _check_langgraph():
            raise ImportError(
                "langgraph is required for EntropyCheckpointer. "
                "Install with: pip install propagul[langgraph]"
            )
        if cls._real_class is None:
            cls._real_class = _build_checkpointer_class()
        return cls._real_class(*args, **kwargs)
