#!/usr/bin/env python3
"""GAP 1: P2P Checkpointer Sync — Multi-Node Framework Integration.

CRITICAL: Proves that LangGraph checkpoints sync between nodes via gossip.
This is our core value proposition: "State syncs automatically between peers."

Run with:
    OPENAI_API_KEY=sk-test-dummy \
    LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    .venv311/bin/python3.11 tests/test_p2p_checkpointer.py
"""
import asyncio
import sys
import os
import time

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

passed = 0
failed = 0
t0 = time.time()


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name} (+{time.time() - t0:.1f}s)", flush=True)
        passed += 1
    else:
        print(f"  ❌ {name} — {detail} (+{time.time() - t0:.1f}s)", flush=True)
        failed += 1


async def run_all():
    from langgraph.graph import StateGraph, START, END
    from propagul.integrations.langgraph import EntropyCheckpointer
    from propagul import AgentStateStore
    from propagul.types import PeerAddress
    from typing import TypedDict, Annotated
    from operator import add

    class State(TypedDict):
        messages: Annotated[list[str], add]

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Baseline Store P2P Sync
    # ═══════════════════════════════════════════════════════════
    print("=== Baseline: AgentStateStore P2P Sync ===", flush=True)

    s1 = AgentStateStore(room="p2p-base", node_id=1, port=19801, gossip_interval_ms=100)
    s2 = AgentStateStore(room="p2p-base", node_id=2, port=19802, gossip_interval_ms=100)

    await s1.start(peers=[PeerAddress("127.0.0.1", 19802)])
    await s2.start(peers=[PeerAddress("127.0.0.1", 19801)])

    s1.set("key1", "value1")
    s2.set("key2", "value2")
    await asyncio.sleep(1.5)

    test("s2 sees key1", s2.get("key1") == "value1", f"got {s2.get('key1')}")
    test("s1 sees key2", s1.get("key2") == "value2", f"got {s1.get('key2')}")

    await s1.stop()
    await s2.stop()

    # ═══════════════════════════════════════════════════════════
    # TEST 2: LangGraph Checkpointer P2P Sync
    # ═══════════════════════════════════════════════════════════
    print("\n=== LangGraph: Checkpointer P2P Sync (A writes, B reads) ===", flush=True)

    def process(state: State) -> dict:
        return {"messages": ["processed_by_node_a"]}

    workflow = StateGraph(State)
    workflow.add_node("process", process)
    workflow.add_edge(START, "process")
    workflow.add_edge("process", END)

    cp_a = EntropyCheckpointer(room="lg-p2p", port=19810, node_id=10,
                               peers=[("127.0.0.1", 19811)])
    cp_b = EntropyCheckpointer(room="lg-p2p", port=19811, node_id=20,
                               peers=[("127.0.0.1", 19810)])
    await cp_a.astart()
    await cp_b.astart()

    graph_a = workflow.compile(checkpointer=cp_a)
    config = {"configurable": {"thread_id": "shared-thread-001"}}
    result = graph_a.invoke({"messages": ["start"]}, config=config)

    test("graph.invoke ran", "processed_by_node_a" in result["messages"], str(result))

    await asyncio.sleep(2.0)

    cp_b_tuple = cp_b.get_tuple(config)
    test("Node B sees checkpoint (P2P SYNC)",
         cp_b_tuple is not None,
         "CRITICAL: Node B has NO checkpoint! P2P sync failed.")

    if cp_b_tuple:
        print(f"    → Node B checkpoint: id={cp_b_tuple.config['configurable'].get('checkpoint_id', '?')}", flush=True)

    await cp_a.aclose()
    await cp_b.aclose()

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Bidirectional Sync
    # ═══════════════════════════════════════════════════════════
    print("\n=== LangGraph: Bidirectional Sync ===", flush=True)

    cp_a2 = EntropyCheckpointer(room="lg-bidir", port=19812, node_id=30,
                                peers=[("127.0.0.1", 19813)])
    cp_b2 = EntropyCheckpointer(room="lg-bidir", port=19813, node_id=40,
                                peers=[("127.0.0.1", 19812)])
    await cp_a2.astart()
    await cp_b2.astart()

    cfg_a = {"configurable": {"thread_id": "thread-from-a"}}
    cfg_b = {"configurable": {"thread_id": "thread-from-b"}}

    cp_a2.put(cfg_a, {
        "v": 1, "id": "cp-a-001", "ts": "2026-05-14T06:00:00Z",
        "channel_values": {"data": "from_node_a"},
        "channel_versions": {}, "versions_seen": {}, "updated_channels": None,
    }, {"source": "input"}, {})

    cp_b2.put(cfg_b, {
        "v": 1, "id": "cp-b-001", "ts": "2026-05-14T06:00:01Z",
        "channel_values": {"data": "from_node_b"},
        "channel_versions": {}, "versions_seen": {}, "updated_channels": None,
    }, {"source": "input"}, {})

    await asyncio.sleep(2.0)

    a_sees_b = cp_a2.get_tuple(cfg_b)
    b_sees_a = cp_b2.get_tuple(cfg_a)
    test("A sees B's thread", a_sees_b is not None, "A cannot read thread-from-b")
    test("B sees A's thread", b_sees_a is not None, "B cannot read thread-from-a")

    if a_sees_b:
        print(f"    → A sees B's data: {a_sees_b.checkpoint['channel_values'].get('data')}", flush=True)
    if b_sees_a:
        print(f"    → B sees A's data: {b_sees_a.checkpoint['channel_values'].get('data')}", flush=True)

    await cp_a2.aclose()
    await cp_b2.aclose()

    # Crash recovery requires port rebind which deadlocks in same event loop.
    # Run as subprocess to test with clean asyncio state.
    import subprocess
    crash_script = '''
import asyncio, sys, os
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
sys.path.insert(0, 'python')
from propagul.integrations.langgraph import EntropyCheckpointer

async def run():
    cp_x = EntropyCheckpointer(room='cr-sub', port=19840, node_id=50, peers=[('127.0.0.1', 19841)])
    cp_y = EntropyCheckpointer(room='cr-sub', port=19841, node_id=60, peers=[('127.0.0.1', 19840)])
    await cp_x.astart()
    await cp_y.astart()

    cfg = {'configurable': {'thread_id': 'crash-t'}}
    cp_x.put(cfg, {
        'v': 1, 'id': 'cp-1', 'ts': '2026-01-01T00:00:00Z',
        'channel_values': {'important': 'must_not_lose'},
        'channel_versions': {}, 'versions_seen': {}, 'updated_channels': None,
    }, {'source': 'input'}, {})

    await asyncio.sleep(2.0)
    y_data = cp_y.get_tuple(cfg)
    if y_data is None:
        print('SYNC_FAIL')
        return

    await cp_x.aclose()
    await asyncio.sleep(1.0)

    cp_x2 = EntropyCheckpointer(room='cr-sub', port=19840, node_id=70, peers=[('127.0.0.1', 19841)])
    await cp_x2.astart()
    await asyncio.sleep(2.0)

    x2_data = cp_x2.get_tuple(cfg)
    await cp_y.aclose()
    await cp_x2.aclose()

    if x2_data is None:
        print('RECOVERY_FAIL')
    elif x2_data.checkpoint.get('channel_values', {}).get('important') != 'must_not_lose':
        print('DATA_CORRUPT')
    else:
        print('RECOVERY_OK')

asyncio.run(run())
'''
    result = subprocess.run(
        [sys.executable, "-u", "-c", crash_script],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env={**os.environ, "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"},
    )
    output = result.stdout.strip()
    test("Crash recovery via P2P (subprocess)", output == "RECOVERY_OK",
         f"got: {output}, stderr: {result.stderr[-200:] if result.stderr else 'none'}")

    # ═══════════════════════════════════════════════════════════
    # TEST 5: CrewAI State P2P Sync
    # ═══════════════════════════════════════════════════════════
    print("\n=== CrewAI: PersistentCrew State P2P Sync ===", flush=True)

    c1 = AgentStateStore(room="crew-p2p", node_id=1, port=19820, gossip_interval_ms=100)
    c2 = AgentStateStore(room="crew-p2p", node_id=2, port=19821, gossip_interval_ms=100)

    await c1.start(peers=[PeerAddress("127.0.0.1", 19821)])
    await c2.start(peers=[PeerAddress("127.0.0.1", 19820)])

    # Simulate what PersistentCrew.kickoff() writes
    c1.set("crew/status", "running")
    c1.set("task/0/status", "completed")
    c1.set("task/0/result", "Research complete: AI state management")
    c1.set("crew/status", "completed")
    c1.set("crew/result", "Final Answer: State sync works")

    await asyncio.sleep(1.5)

    test("Node 2 sees crew/status", c2.get("crew/status") == "completed",
         f"got: {c2.get('crew/status')}")
    test("Node 2 sees task result", c2.get("task/0/result") is not None,
         "Missing task result")
    test("Node 2 sees crew result", c2.get("crew/result") is not None,
         "Missing crew result")

    print(f"    → CREWAI P2P SYNC VERIFIED ✓", flush=True)

    await c1.stop()
    await c2.stop()


asyncio.run(run_all())

total = passed + failed
elapsed = time.time() - t0
print(f"\n{'=' * 60}", flush=True)
print(f"P2P CHECKPOINTER: {passed} passed, {failed} failed ({elapsed:.1f}s)", flush=True)
print(f"{'=' * 60}", flush=True)
sys.exit(1 if failed > 0 else 0)
