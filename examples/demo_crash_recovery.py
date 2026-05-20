#!/usr/bin/env python3
"""Propagul Killer Demo: "Agents work through network failure."

3 agents collaborate on a task list. Mid-way through, Agent C "crashes"
(network disconnect). Agents A and B continue working. When Agent C
reconnects, it automatically recovers ALL state — zero data loss.

Runtime: ~8 seconds. No external dependencies. No API keys.
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from propagul.store import AgentStateStore
from propagul.types import PeerAddress

# ANSI colors for visual clarity
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

TASKS = [
    ("task.1.research",    "Market Analysis"),
    ("task.2.prototype",   "Build MVP"),
    ("task.3.tests",       "Write Tests"),
    ("task.4.deploy",      "Deploy to Prod"),
    ("task.5.docs",        "Write Documentation"),
    ("task.6.launch",      "Public Launch"),
]


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}{RESET}\n")


def status(agent: str, msg: str, color: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  {DIM}{ts}{RESET}  {color}{agent}{RESET}  {msg}")


async def main():
    banner("PROPAGUL DEMO: Agents Work Through Network Failure")

    # ─── Phase 1: Create 3 agents ─────────────────────────────────

    print(f"  {CYAN}Creating 3 agents with P2P gossip sync...{RESET}")

    agent_a = AgentStateStore(room="demo", node_id=1, port=9300, host="127.0.0.1")
    agent_b = AgentStateStore(room="demo", node_id=2, port=9301, host="127.0.0.1")
    agent_c = AgentStateStore(room="demo", node_id=3, port=9302, host="127.0.0.1")

    peers_all = [
        PeerAddress("127.0.0.1", 9300),
        PeerAddress("127.0.0.1", 9301),
        PeerAddress("127.0.0.1", 9302),
    ]

    await agent_a.start(peers=peers_all)
    await agent_b.start(peers=peers_all)
    await agent_c.start(peers=peers_all)

    status("Agent A", "online (node_id=1, port=9300)", GREEN)
    status("Agent B", "online (node_id=2, port=9301)", GREEN)
    status("Agent C", "online (node_id=3, port=9302)", GREEN)

    # ─── Phase 2: All agents work together ────────────────────────

    banner("Phase 1: Collaborative Work (all connected)")

    # Agent A completes task 1
    agent_a.set("task.1.research", "DONE")
    status("Agent A", f"completed: {TASKS[0][1]}", GREEN)
    await asyncio.sleep(0.8)  # Let gossip sync

    # Agent B completes task 2
    agent_b.set("task.2.prototype", "DONE")
    status("Agent B", f"completed: {TASKS[1][1]}", GREEN)
    await asyncio.sleep(0.8)

    # Verify sync
    a_keys = len([k for k in ["task.1.research", "task.2.prototype"]
                  if agent_c.get(k) == "DONE"])
    status("Agent C", f"synced {a_keys}/2 tasks from peers ✓", CYAN)

    # ─── Phase 3: Agent C crashes ─────────────────────────────────

    banner("Phase 2: NETWORK FAILURE — Agent C Disconnects")

    await agent_c.stop()
    status("Agent C", "💥 CRASHED (network down)", RED)
    print(f"\n  {YELLOW}Agents A and B continue working...{RESET}\n")
    await asyncio.sleep(0.3)

    # Agents A and B keep working
    agent_a.set("task.3.tests", "DONE")
    status("Agent A", f"completed: {TASKS[2][1]}", GREEN)
    await asyncio.sleep(0.8)

    agent_b.set("task.4.deploy", "DONE")
    status("Agent B", f"completed: {TASKS[3][1]}", GREEN)
    await asyncio.sleep(0.8)

    agent_a.set("task.5.docs", "DONE")
    status("Agent A", f"completed: {TASKS[4][1]}", GREEN)
    await asyncio.sleep(1.0)  # Extra time for A↔B full sync

    # Show that A and B are synced but C missed everything
    a_done = sum(1 for t, _ in TASKS if agent_a.get(t) == "DONE")
    b_done = sum(1 for t, _ in TASKS if agent_b.get(t) == "DONE")
    print()
    status("Agent A", f"knows {a_done}/6 tasks completed", GREEN)
    status("Agent B", f"knows {b_done}/6 tasks completed", GREEN)
    status("Agent C", "offline — missed 3 tasks", RED)

    # ─── Phase 4: Agent C reconnects ──────────────────────────────

    banner("Phase 3: RECOVERY — Agent C Reconnects")

    # Restart Agent C — new instance, same room
    agent_c = AgentStateStore(room="demo", node_id=3, port=9302, host="127.0.0.1")
    await agent_c.start(peers=peers_all)
    status("Agent C", "reconnected!", YELLOW)

    # Wait for gossip sync
    print(f"\n  {CYAN}Waiting for automatic state recovery...{RESET}")
    await asyncio.sleep(2.5)  # ~5 gossip rounds for reliable convergence

    # ─── Phase 5: Verify recovery ─────────────────────────────────

    banner("Phase 4: VERIFICATION — Zero Data Loss")

    c_done = sum(1 for t, _ in TASKS[:5] if agent_c.get(t) == "DONE")

    for task_key, task_name in TASKS[:5]:
        val = agent_c.get(task_key)
        icon = f"{GREEN}✅{RESET}" if val == "DONE" else f"{RED}❌{RESET}"
        print(f"  {icon}  {task_name}: {val or 'MISSING'}")

    # Agent C completes the last task (proving it's fully operational)
    agent_c.set("task.6.launch", "DONE")
    status("Agent C", f"completed: {TASKS[5][1]} (agent fully recovered!)", GREEN)
    await asyncio.sleep(1.5)  # Let C's write propagate to A and B

    # Final state
    print()
    for name, agent in [("Agent A", agent_a), ("Agent B", agent_b), ("Agent C", agent_c)]:
        done = sum(1 for t, _ in TASKS if agent.get(t) == "DONE")
        color = GREEN if done == 6 else YELLOW
        status(name, f"{done}/6 tasks — {'ALL SYNCED ✓' if done == 6 else 'syncing...'}", color)

    # ─── Summary ──────────────────────────────────────────────────

    all_synced = all(
        agent.get(t) == "DONE"
        for agent in [agent_a, agent_b, agent_c]
        for t, _ in TASKS
    )

    banner("RESULT")
    if all_synced:
        print(f"  {GREEN}{BOLD}✅ ZERO DATA LOSS{RESET}")
        print(f"  {GREEN}   All 3 agents have all 6 tasks.{RESET}")
        print(f"  {GREEN}   Agent C recovered automatically after crash.{RESET}")
        print(f"  {GREEN}   No manual intervention. No database. No server.{RESET}")
    else:
        print(f"  {YELLOW}⚠️  Partial recovery (gossip still converging){RESET}")

    print(f"\n  {DIM}Protocol: OR-Map CRDT + Push-Pull Gossip (TCP)")
    print(f"  Dependencies: None (pure Python)")
    print(f"  External services: None (peer-to-peer){RESET}\n")

    # Cleanup
    await agent_a.stop()
    await agent_b.stop()
    await agent_c.stop()


if __name__ == "__main__":
    asyncio.run(main())
