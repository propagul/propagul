#!/usr/bin/env python3
"""System E2E Test — proves entropy-state works end-to-end.

Tests the FULL stack:
1. Core: EntropyAgent + StateMap (Rust)
2. SDK: AgentStateStore + GossipTransport (Python)
3. Server: RoomManager + Auth + Persistence (Python)
4. Cloud Connect: SDK → Server integration

This is the definitive "does it work?" test.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from entropy_state.types import PeerAddress

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1


async def main():
    # ── Test 1: Core works ──────────────────────────────────────────
    print("\n=== 1. Core (Rust) ===")
    from entropy_state_core import EntropyAgent, StateMap

    agent = EntropyAgent(1)
    check("EntropyAgent created", agent.entropy >= 0)

    state = StateMap(1)
    state.set("key", "value")
    check("StateMap set/get", state.get("key") == "value")

    snap = state.snapshot()
    check("snapshot serializes", len(snap) > 0)

    state2 = StateMap(2)
    delta = state2.merge(snap)
    check("merge works", state2.get("key") == "value")
    check("merge delta > 0", delta > 0)

    # ── Test 2: SDK works (3-agent gossip) ──────────────────────────
    print("\n=== 2. SDK (3-Agent Gossip) ===")
    from entropy_state import AgentStateStore

    s1 = AgentStateStore(room="e2e-test", node_id=1, port=19301)
    s2 = AgentStateStore(room="e2e-test", node_id=2, port=19302)
    s3 = AgentStateStore(room="e2e-test", node_id=3, port=19303)

    peers12 = [("127.0.0.1", 19302), ("127.0.0.1", 19303)]
    peers23 = [("127.0.0.1", 19301), ("127.0.0.1", 19303)]
    peers31 = [("127.0.0.1", 19301), ("127.0.0.1", 19302)]

    await s1.start(peers=peers12)
    await s2.start(peers=peers23)
    await s3.start(peers=peers31)
    check("3 agents started", s1.is_running and s2.is_running and s3.is_running)

    # Agent 1 sets state
    s1.set("mission", "research-ai")
    s1.set("status", "active")

    # Wait for convergence
    await asyncio.sleep(3)

    check("s2 received mission", s2.get("mission") == "research-ai")
    check("s3 received mission", s3.get("mission") == "research-ai")
    check("s2 received status", s2.get("status") == "active")

    # Agent 2 modifies state
    s2.set("status", "completed")
    await asyncio.sleep(2)
    check("s1 sees update", s1.get("status") == "completed")
    check("s3 sees update", s3.get("status") == "completed")

    # Delete propagation
    s3.delete("mission")
    await asyncio.sleep(2)
    check("delete propagated to s1", s1.get("mission") is None)
    check("delete propagated to s2", s2.get("mission") is None)

    # Stats
    stats = s1.stats
    check("entropy >= 0", stats.entropy >= 0)
    check("gossip_rounds > 0", stats.gossip_rounds > 0)

    await s1.stop()
    await s2.stop()
    await s3.stop()
    check("all stopped", not s1.is_running)

    # ── Test 3: Crash Recovery ──────────────────────────────────────
    print("\n=== 3. Crash Recovery ===")

    # Agent A has state
    sa = AgentStateStore(room="crash-test", node_id=10, port=19310)
    sb = AgentStateStore(room="crash-test", node_id=11, port=19311)

    await sa.start(peers=[("127.0.0.1", 19311)])
    await sb.start(peers=[("127.0.0.1", 19310)])

    sa.set("task_0_status", "done")
    sa.set("task_0_result", "42")
    await asyncio.sleep(2)

    check("sb has task_0 before crash", sb.get("task_0_status") == "done")

    # "Crash" sb
    await sb.stop()

    # sa continues working
    sa.set("task_1_status", "done")
    sa.set("task_1_result", "99")

    # "Restart" sb (new instance, same room)
    sb_new = AgentStateStore(room="crash-test", node_id=11, port=19311)
    await sb_new.start(peers=[("127.0.0.1", 19310)])
    await asyncio.sleep(3)

    check("recovered task_0", sb_new.get("task_0_status") == "done")
    check("recovered task_0 result", sb_new.get("task_0_result") == "42")
    check("got task_1 (set during crash)", sb_new.get("task_1_status") == "done")
    check("got task_1 result", sb_new.get("task_1_result") == "99")

    await sa.stop()
    await sb_new.stop()

    # ── Test 4: Server (RoomManager + Auth + Persistence) ───────────
    print("\n=== 4. Server Components ===")
    from entropy_state.server.room_manager import RoomManager
    from entropy_state.server.auth import AuthManager, TIER_PRO
    from entropy_state.server.persistence import save_room_state, load_room_state

    # Auth
    auth = AuthManager()
    key = auth.add_key("es_e2e_test", tier=TIER_PRO, owner="e2e@test.com")
    check("auth: key created", key.max_rooms == 10)

    validated = auth.validate("Bearer es_e2e_test")
    check("auth: validates correctly", validated is not None)

    # RoomManager
    mgr = RoomManager(port_start=19320, port_end=19325)
    info = await mgr.create_room("e2e-room", "es_e2e_test")
    check("room created", info.room_id == "e2e-room")

    store = mgr.get_store("e2e-room")
    store.set("server_data", "persisted")
    state_dict = mgr.get_room_state("e2e-room")
    check("server state accessible", state_dict.get("server_data") == "persisted")

    # Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_room_state(tmpdir, "e2e-room", state_dict, info.to_dict())
        loaded = load_room_state(path)
        check("persistence roundtrip", loaded[1]["server_data"] == "persisted")

    await mgr.delete_room("e2e-room")
    check("room deleted", mgr.room_count == 0)
    await mgr.stop()

    # ── Test 5: SDK → Server Integration ────────────────────────────
    print("\n=== 5. SDK ↔ Server Integration ===")

    mgr2 = RoomManager(port_start=19330, port_end=19335)
    server_room = await mgr2.create_room("cloud-room", "es_key")
    server_store = mgr2.get_store("cloud-room")

    # Client agent connects to server's gossip port
    client = AgentStateStore(room="cloud-room", node_id=1, port=19340)
    await client.start(peers=[("127.0.0.1", server_room.port)])

    # Server must also know about client (bidirectional gossip)
    # In production, connect_to_cloud() handles this via the HTTP API
    server_store.add_peer(PeerAddress("127.0.0.1", 19340))

    # Client writes
    client.set("agent_state", "working")
    await asyncio.sleep(2)

    # Server should have the state
    check("server got client state", server_store.get("agent_state") == "working")

    # Server writes back
    server_store.set("server_ack", "received")
    await asyncio.sleep(2)

    # Client should have the ack
    check("client got server ack", client.get("server_ack") == "received")

    # Simulate client crash + restart
    await client.stop()

    server_store.set("while_offline", "data_saved")

    client2 = AgentStateStore(room="cloud-room", node_id=1, port=19340)
    await client2.start(peers=[("127.0.0.1", server_room.port)])
    await asyncio.sleep(3)

    check("after restart: agent_state recovered", client2.get("agent_state") == "working")
    check("after restart: server_ack recovered", client2.get("server_ack") == "received")
    check("after restart: offline data synced", client2.get("while_offline") == "data_saved")

    await client2.stop()
    await mgr2.stop()


asyncio.run(main())

print(f"\n{'='*60}")
print(f"SYSTEM E2E: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
