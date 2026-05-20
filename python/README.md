# entropy-state

**Zero-Ops crash-resilient state sync for multi-agent AI systems.**

> Your agents crash. Their state shouldn't die with them.

## The Problem

AI agent frameworks (LangGraph, CrewAI) use simple checkpoints — they save state to a database,
but they don't prevent **duplicate execution** on restart. When a crash occurs mid-workflow:

- The agent re-executes expensive LLM calls ($2+ per retry)
- No distributed locking → concurrent restarts corrupt state
- Recovery requires manual `thread_id` lookup and re-invocation
- Traditional durable execution (Temporal) requires massive infrastructure

## The Solution

`entropy-state` replaces fragile checkpoints with **CRDT-based peer-to-peer state sync**:

- **Automatic Crash Recovery**: Agent dies → restarts → recovers state from peers. No manual intervention.
- **Zero Infrastructure**: No database, no message broker, no Kubernetes. Agents sync directly via TCP gossip.
- **Conflict Resolution**: Concurrent writes from multiple agents are preserved (OR-Map CRDT), not silently overwritten.
- **48% Packet Loss Tolerance**: Adaptive gossip protocol (calibrated over 218,700 simulations) converges even on unreliable networks.
- **Drop-in Plugins**: Works with CrewAI and LangGraph — replaces their checkpointers, not their API.

## Install

```bash
pip install entropy-state                  # Core + SDK
pip install entropy-state[crewai]          # + CrewAI PersistentCrew
pip install entropy-state[langgraph]       # + LangGraph EntropyCheckpointer
pip install entropy-state[server]          # + Managed Sync Service
pip install entropy-state[all]             # Everything
```

## Quick Start

### Standalone (no framework)

```python
import asyncio
from entropy_state import AgentStateStore
from entropy_state.types import PeerAddress

async def main():
    # Agent 1
    store = AgentStateStore(room="my-project", node_id=1, port=9001)
    store.set("task", "research")
    store.set("status", "running")

    # Start gossip sync
    await store.start(peers=[PeerAddress("127.0.0.1", 9002)])

    # State is now syncing with peers...
    print(store.get("task"))         # "research"
    print(store.stats.entropy)       # Current entropy estimate
    print(store.stats.gossip_rounds) # Number of sync rounds

    await store.stop()

asyncio.run(main())
```

### With CrewAI

Upgrades CrewAI from local-only memory to **distributed, crash-resilient state** across multiple machines.
On crash + restart, previously completed tasks are skipped — no duplicate LLM calls.

```python
from crewai import Agent, Task
from entropy_state.integrations.crewai import PersistentCrew

researcher = Agent(role="Researcher", goal="Research AI trends", backstory="Expert")
task = Task(description="Research entropy-state applications", expected_output="A report", agent=researcher)

# Drop-in replacement for crewai.Crew — state persists across crashes
crew = PersistentCrew(
    agents=[researcher],
    tasks=[task],
    state_room="my-project",    # Logical room for state sync
    state_port=9001,            # TCP port for gossip
    recovery=True,              # Auto-recover state from peers on restart
)

result = crew.kickoff()
```

### With LangGraph

Replaces `PostgresSaver`/`SqliteSaver` with a **lock-free, CRDT-backed checkpointer**.
No central database needed — checkpoints sync via gossip between graph instances.

```python
from langgraph.graph import StateGraph
from entropy_state.integrations.langgraph import EntropyCheckpointer

checkpointer = EntropyCheckpointer(room="my-graph", port=9001)

workflow = StateGraph(...)
graph = workflow.compile(checkpointer=checkpointer)

result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})
```

## API Reference

### `AgentStateStore` — Main API

```python
from entropy_state import AgentStateStore

store = AgentStateStore(
    room="my-project",           # Logical namespace
    node_id=1,                   # Unique ID per node (required)
    port=9000,                   # TCP gossip port
    host="0.0.0.0",              # Bind address
    gossip_interval_ms=500.0,    # Gossip frequency
    gossip_secret=b"my-secret",  # Optional: HMAC auth for gossip
)
```

| Method | Description |
|--------|-------------|
| `store.set(key, value)` | Set a key-value pair |
| `store.get(key) → str\|None` | Get current value |
| `store.get_all() → dict` | Get all key-value pairs |
| `store.delete(key)` | Delete key (propagates via tombstone) |
| `store.get_conflicts(key) → list` | Get concurrent values (conflict detection) |
| `store.on_change(callback)` | Register change listener |
| `await store.start(peers=[...])` | Start gossip sync |
| `await store.stop()` | Stop gossip sync |
| `store.stats` | Telemetry (entropy, gossip rounds, peers, etc.) |

### `PersistentCrew` — CrewAI Plugin

Drop-in replacement for `crewai.Crew`. Auto-persists task status + results.
On crash + restart, previously completed tasks are recovered from peers.

```python
from entropy_state.integrations.crewai import PersistentCrew
```

### `EntropyCheckpointer` — LangGraph Plugin

Implements `BaseCheckpointSaver`. Replaces SQLite/Postgres with CRDT gossip.

```python
from entropy_state.integrations.langgraph import EntropyCheckpointer
```

| Method | Description |
|--------|-------------|
| `put(config, checkpoint, metadata)` | Store checkpoint |
| `get_tuple(config)` | Retrieve latest checkpoint |
| `list(config)` | List checkpoint history |
| `put_writes(config, writes, task_id)` | Store pending writes |
| `delete_thread(thread_id)` | Delete all checkpoints for a thread |

## How It Compares

|  | Standard Checkpointers (LangGraph/CrewAI) | Traditional Durable Execution (Temporal) | **entropy-state** |
|---|---|---|---|
| **Infrastructure** | PostgreSQL/SQLite | Temporal Server + DB + Workers | **None** (P2P) |
| **Crash recovery** | Manual (find thread_id, re-invoke) | Automatic (deterministic replay) | **Automatic** (CRDT merge) |
| **Duplicate prevention** | None | Yes (event sourcing) | **Yes** (idempotent CRDT merge) |
| **Multi-agent sync** | No (single-process) | Via workflows | **Yes** (gossip protocol) |
| **Conflict resolution** | Last-write-wins | Deterministic replay | **All values preserved** (OR-Map) |
| **Packet loss tolerance** | 0% | 0% | **48%** |
| **Setup time** | 5 min (DB config) | Hours (cluster setup) | **30 seconds** (`pip install`) |

## Architecture

```
┌─────────────────────────────────┐
│  Your Code (CrewAI / LangGraph) │
├─────────────────────────────────┤
│  entropy-state (Python SDK)     │  ← Open source, pip install
├─────────────────────────────────┤
│  entropy-state-core (Rust)      │  ← Compiled binary, IP-protected
│  • EntropyAgent (adaptive gossip)│
│  • StateMap (OR-Map CRDT)       │
└─────────────┬───────────────────┘
              │ TCP Gossip (HMAC-authenticated)
              ▼
┌─────────────────────────────────┐
│  Other Agents (same room)       │
│  Crash recovery via CRDT merge  │
└─────────────────────────────────┘
```

## License

Core algorithm (Rust): Proprietary — distributed as compiled binary only.
Python SDK: MIT.
