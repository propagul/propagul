# A2A State Layer — Architecture & API Reference

> CRDT-backed shared state for multi-agent AI workflows.

## Overview

The A2A (Agent-to-Agent) State Layer extends Propagul's CRDT infrastructure
to enable **decentralized state sharing between AI agents**. Any agent — local
GPU, cloud API, or edge device — can join a "room" and share key-value state
that converges automatically without central coordination.

```
Agent A (Local GPU) ──→ SharedState ──→ Agent B (Cloud API)
                           ↕ CRDT Sync
                      Agent C (Edge)
```

## Core Concepts

### SharedAgentState

The primary abstraction. Wraps an ORMap CRDT with three namespaces:

| Namespace | Pattern | Access | Use Case |
|-----------|---------|--------|----------|
| `shared:` | `shared:{key}` | Any agent | Coordination state |
| `agent:` | `agent:{id}:{key}` | Owner writes, all read | Private state |
| `meta:` | `meta:{key}` | System managed | Room metadata, presence |

### AgentRoom

Manages lifecycle and membership for a collaboration context:
- **Join/Leave** — Agents register with capabilities
- **State** — Shared state auto-created on first join
- **Presence** — Automatic heartbeat via CRDT timestamps

### Conflict Resolution

**Last-Writer-Wins (LWW)** via CRDT vector clocks. Concurrent writes
to the same key resolve deterministically. The ORMap guarantees:
- **Commutative**: merge(A, B) = merge(B, A)
- **Associative**: merge(merge(A, B), C) = merge(A, merge(B, C))
- **Idempotent**: merge(A, A) = A

## API Reference

### SharedAgentState

```python
from propagul.a2a import SharedAgentState

state = SharedAgentState(room="project-alpha", agent_id="agent-01")

# Shared state (visible to all agents)
state.set("task_status", "processing")
state.set("progress", "0.75")
result = state.get("task_status")     # "processing"
all_shared = state.get_all_shared()   # {"task_status": "processing", ...}
state.delete("progress")

# Private state (per-agent namespace)
state.set_private("internal_cache", "some_data")
my_val = state.get_private("internal_cache")

# Read another agent's state
other = state.get_agent_state("agent-02")  # dict

# Room metadata
state.set_meta("description", "Collaborative RAG pipeline")
state.get_meta("description")

# CRDT sync
snapshot = state.snapshot()       # For wire transmission
result = state.merge(remote)     # Returns SyncResult
```

### AgentRoom

```python
from propagul.a2a import AgentRoom, A2AConfig

config = A2AConfig(persist_path="/data/rooms")
room = AgentRoom("project-alpha", config=config)

# Membership
info = room.join("agent-01", capabilities=["inference", "rag"])
room.join("agent-02", capabilities=["search"])
room.has_member("agent-01")   # True
room.leave("agent-02")

# Access state
state = room.state
state.set("query", "What is CRDT?")

# Introspection
members = room.members()        # List[AgentInfo]
active = room.active_members()  # Filtered by TTL
summary = room.summary()
```

## Use Cases

### 1. Distributed RAG Pipeline

```
Agent-Retriever  →  shared:documents = "[doc1, doc2]"
Agent-Ranker     →  shared:ranked = "[doc2, doc1]"
Agent-Generator  →  shared:answer = "CRDT stands for..."
```

### 2. Swarm Intelligence (Multi-Model Voting)

```
Agent-GPT4    →  agent:gpt4:vote = "option_a"
Agent-Claude  →  agent:claude:vote = "option_b"
Agent-Llama   →  agent:llama:vote = "option_a"
Orchestrator  →  shared:consensus = "option_a" (2/3 majority)
```

### 3. Fleet-Wide Inference Coordination

```
Node-1  →  agent:n1:load = "0.85"
Node-2  →  agent:n2:load = "0.30"
Router  →  shared:next_target = "node-2" (lowest load)
```

## Architecture

```
propagul/a2a/
├── __init__.py    — Public exports
├── types.py       — A2AConfig, AgentInfo, SyncResult
├── state.py       — SharedAgentState (ORMap wrapper)
└── room.py        — AgentRoom (membership + lifecycle)
```

### Persistence

Optional. Reuses `config_sync.py` patterns:
- Atomic writes (tmp + os.replace)
- Debounced I/O (default 2s)
- JSON format, version-tagged
- Merge-on-load (not overwrite)

### Transport

Phase 1 (current): HTTP push/pull via dashboard API.
Phase 2 (planned): Direct P2P gossip via `propagul.gossip`.

## Limitations (Phase 1)

1. **No real-time push**: Agents poll for updates (10s heartbeat cycle)
2. **String values only**: Complex data must be JSON-serialized by caller
3. **No access control**: Any agent in a room can write to `shared:`
4. **No cross-room sync**: Rooms are isolated
5. **Single-process**: No distributed lock or transaction support

## Security Considerations

- Room state is **encrypted at rest** if PROPAGUL_KEY_SALT is set
- Agent IDs are **not validated** — caller is responsible for uniqueness
- State snapshots transmitted over wire must use TLS
- **No PII storage**: Agent state should not contain personal data

---

*Version: 0.12.0-dev · Phase 3.9*
