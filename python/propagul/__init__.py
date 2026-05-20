"""propagul — Zero-config P2P state sync for AI agents.

Pure Python, crash-resilient, conflict-free replicated state.
No external dependencies. No compilation required.

Usage:
    from propagul import AgentStateStore

    store = AgentStateStore(room="my-project", node_id=1, port=9001)
    store.set("task", "research")

    # Start gossip sync with peers
    await store.start(peers=[("other-agent", 9002)])

    # Optional: connect to cloud peer for persistence
    store.connect_to_cloud(api_key="pg_live_abc123")
"""

from propagul.store import AgentStateStore
from propagul.crdt import ORMap
from propagul.transport import GossipTransport, create_tls_context
from propagul.discovery import StaticDiscovery

__version__ = "0.7.1"
__all__ = ["AgentStateStore", "ORMap", "GossipTransport", "StaticDiscovery", "create_tls_context"]
