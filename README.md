# Propagul Mesh

> **One dashboard for all your local AI servers.**
> See GPU temps, pull models remotely, route inference — without SSH, Grafana, or LiteLLM.

[![Tests](https://img.shields.io/badge/tests-408%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![PyPI](https://img.shields.io/pypi/v/propagul-mesh)]()

## The Problem

You run Ollama on your desktop, vLLM on a home server, and TGI on a rented GPU.
To check GPU temps, pull a model, or see which node has capacity — you SSH into each one.
When something goes down overnight, you find out the next morning.

**Propagul Mesh** gives you one dashboard for all of them. Zero port-forwarding, zero Docker.

## Quick Start

```bash
# 1. Install
pip install propagul-mesh

# 2. Start the agent (auto-detects Ollama, vLLM, TGI, llama.cpp, LM Studio)
propagul-mesh start --api-key pg_live_your_key

# 3. Open your dashboard
# → https://fleet.propagul.dev
```

Your node appears in the dashboard within 10 seconds. GPU temps, VRAM, loaded models — all live.

## What You Get

| Feature | Description |
|---|---|
| **Fleet Dashboard** | Real-time node grid with VRAM, GPU util, temperature, and power sparklines |
| **Remote Model Pull** | Click a button to `ollama pull llama3.1:8b` on any node — no SSH |
| **Multi-Backend** | Auto-detects Ollama, vLLM, TGI, llama.cpp, LM Studio on each machine |
| **Local Proxy** | `localhost:8787/v1/chat/completions` — one OpenAI-compatible endpoint for all backends |
| **Config Sync** | Declare desired model state centrally. Nodes auto-pull and auto-delete to converge. CRDT-backed. |
| **Health Alerts** | Email notification when a node goes offline (via Resend, 4h cooldown) |
| **Prompt Privacy** | Prompts never leave your network. The cloud sees only telemetry (model names, VRAM, temps). |
| **Zero Port-Forwarding** | Agents connect outbound. No inbound ports, no Tailscale, no VPN. |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Machine(s)                                                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Ollama       │  │ vLLM         │  │ TGI          │          │
│  │ :11434       │  │ :8000        │  │ :8080        │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └─────────────────┼─────────────────┘                   │
│                    ┌──────┴───────┐                              │
│                    │ propagul-mesh│ ← pip install, single binary │
│                    │   Agent      │                              │
│                    │ + Local Proxy│ → localhost:8787/v1          │
│                    └──────┬───────┘                              │
│                           │ Telemetry only (no prompts)          │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTPS (outbound only)
                   ┌────────┴────────┐
                   │ cloud.propagul  │
                   │ .dev            │
                   │                 │
                   │ Fleet Dashboard │
                   │ Sparklines      │
                   │ Model Management│
                   │ Health Alerts   │
                   └─────────────────┘
```

**Key insight:** This is the [Tailscale model](https://tailscale.com/blog/how-tailscale-works/) applied to GPU fleet management.
Control plane in the cloud, data plane (your prompts) stays local.

## Supported Backends

| Backend | Detection | Model Listing | Chat (sync) | Chat (stream) | Status |
|---|---|---|---|---|---|
| **Ollama** | ✅ `/api/version` | ✅ `/api/tags` | ✅ | ✅ | Production |
| **vLLM** | ✅ `/v1/models` | ✅ | ✅ | ✅ | Production |
| **TGI** | ✅ `/info` | ✅ (model_id) | ✅ | ✅ | Production |
| **llama.cpp** | ✅ `/health` | ✅ `/v1/models` | ✅ | ✅ | Production |
| **LM Studio** | ✅ `/v1/models` | ✅ | ✅ | ✅ | Production |

Backend detection is automatic — the agent probes each port and identifies the backend
by its health endpoint signature ([detect.py](python/propagul/mesh/backends/detect.py)).

## Local Proxy

The agent runs a local OpenAI-compatible proxy on port `8787`:

```bash
# List models from all detected backends
curl http://localhost:8787/v1/models

# Chat completion (routed to the right backend automatically)
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client (Python `openai`, LangChain, Cursor, Continue, etc.).
Ollama's native API is automatically translated to OpenAI format.

## Security

| Layer | Implementation |
|---|---|
| **API Authentication** | HMAC-SHA256 key derivation (`PROPAGUL_KEY_SALT`) |
| **Content Security Policy** | Strict CSP with nonce-based script loading |
| **Telemetry Validation** | Depth-limited JSON parsing (MAX_DEPTH=5, MAX_KEYS=500) |
| **Heartbeat Limits** | 1 MB max payload, rate-limited per node |
| **SSRF Protection** | RFC 1918 allowlist for proxy targets (no arbitrary IP routing) |
| **Prompt Privacy** | Prompts are processed locally; cloud receives only metadata |

## Configuration

| Variable | Required | Description |
|---|---|---|
| `PROPAGUL_KEY_SALT` | **Yes** | HMAC secret for API key hashing |
| `PROPAGUL_REDIS_URL` | **Yes** (server) | Redis connection URL |
| `PROPAGUL_GOSSIP_SECRET` | Recommended | Shared secret for gossip authentication |
| `PROPAGUL_RESEND_API_KEY` | No | Resend.com API key for email alerts |
| `PROPAGUL_ALERT_EMAIL` | No | Email address for health alerts |
| `PROPAGUL_PUBLIC_HOST` | No | Public hostname (default: `fleet.propagul.dev`) |
| `PROPAGUL_ALLOW_PLAINTEXT` | No | Set `true` for dev without TLS |

## How It Compares

| Feature | GPUStack | LiteLLM | llama-dash | **Propagul Mesh** |
|---|---|---|---|---|
| GPU Fleet Dashboard | ✅ (Grafana) | ❌ | ✅ | ✅ (built-in) |
| Remote Model Pull | ✅ (HF Catalog) | ❌ | ❌ | ✅ (Ollama) |
| Multi-Backend | ✅ | ✅ (100+ providers) | ❌ (llama.cpp only) | ✅ (5 backends) |
| Zero-Config Install | ❌ (K8s/Docker) | ❌ (config file) | ❌ | **✅ (`pip install` + 1 cmd)** |
| Prompt Privacy | ❌ (centralized) | ❌ (proxy sees all) | ✅ (local) | **✅ (Tailscale model)** |
| CRDT Config Sync | ❌ | ❌ | ❌ | **✅ (unique)** |
| Real-time Sparklines | ✅ (Grafana) | ❌ | ✅ (SSE) | **✅ (Chart.js + SSE)** |
| Cost Tracking | ❌ | ✅ | ✅ | ❌ |

### When to use what

- **GPUStack** — Best for teams with K8s/Docker already running, need HuggingFace catalog integration.
- **LiteLLM** — Best as a centralized API gateway with cost tracking (not a fleet manager).
- **llama-dash** — Best for single-machine llama.cpp monitoring with SQLite logging.
- **Propagul Mesh** — Best if you want `pip install` + one command to monitor multiple machines with different backends, and your prompts must stay local.

---

## Advanced: State Sync SDK

Propagul also includes a P2P state synchronization SDK for AI agent workflows.
This is the original core library — the fleet dashboard was built on top of it.

```python
from propagul import AgentStateStore

store = AgentStateStore(room="my-project", node_id=1, port=9001)
store.set("task", "research")

# Agent crashes → restarts → recovers state from peers
await store.start(peers=[("192.168.1.2", 9002)])
print(store.get("task"))  # "research" — recovered from peer
```

Features: ORMap CRDT, adaptive gossip protocol, crash recovery, conflict detection,
CrewAI and LangGraph integrations. See [docs/](docs/) for full SDK documentation.

## License

MIT — Full source available.
