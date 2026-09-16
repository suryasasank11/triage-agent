"""
M3 runner: run ONE ticket through the branching graph. If the graph escalates,
it PAUSES and asks you (the human) to approve / edit / reject, then resumes.

Usage (PowerShell):
    python run_ticket.py T-1002
Defaults to T-1001.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph

load_dotenv()
ROOT = Path(__file__).resolve().parent
SERVER = str(ROOT / "mcp_server" / "server.py")
TICKETS = ROOT / "data" / "tickets.jsonl"
MODEL = "claude-haiku-4-5-20251001"   # swap to "claude-sonnet-5" for more quality


def load_ticket(ticket_id: str) -> dict:
    with TICKETS.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line)["ticket_id"] == ticket_id:
                return json.loads(line)
    raise SystemExit(f"Ticket {ticket_id} not found")


def _print_state(s: dict):
    print(f"category   : {s.get('category')}  "
          f"(priority={s.get('priority')}, confidence={s.get('confidence')})")
    if s.get("review_verdict"):
        print(f"review     : {s.get('review_verdict')} - {s.get('review_notes')}")
    print(f"attempts   : {s.get('attempts')}")


def human_prompt(payload: dict) -> dict:
    print("\n" + "=" * 60)
    print("  HUMAN-IN-THE-LOOP: this ticket was escalated")
    print("=" * 60)
    print(f"  reason     : {payload['reason']}")
    print(f"  category   : {payload['category']}  (confidence={payload['confidence']})")
    print(f"\n  draft on the table:\n  {payload['draft']}")
    print("-" * 60)
    while True:
        choice = input("  [a]pprove / [e]dit / [r]eject ? ").strip().lower()
        if choice in ("a", "approve"):
            return {"decision": "approve", "body": ""}
        if choice in ("e", "edit"):
            body = input("  new reply text: ").strip()
            return {"decision": "edit", "body": body}
        if choice in ("r", "reject"):
            return {"decision": "reject"}


async def main() -> None:
    ticket_id = sys.argv[1] if len(sys.argv) > 1 else "T-1001"
    ticket = load_ticket(ticket_id)

    client = MultiServerMCPClient({
        "harbor": {"command": sys.executable, "args": [SERVER], "transport": "stdio"}})
    tools = await client.get_tools()
    model = ChatAnthropic(model=MODEL, temperature=0, max_tokens=1024)
    graph = build_graph(tools, model, InMemorySaver())
    cfg = {"configurable": {"thread_id": ticket_id}}

    print(f"\n>>> {ticket_id}  |  {ticket['subject']}  (from {ticket['sender']})\n")
    result = await graph.ainvoke({"ticket": ticket}, cfg)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        _print_state(result)
        decision = human_prompt(payload)
        result = await graph.ainvoke(Command(resume=decision), cfg)

    print("\n--- outcome ---")
    _print_state(result)
    print(f"final_action : {result.get('final_action')}")
    if result.get("final_action") in ("auto_sent", "human_approved"):
        print(f"\n--- reply sent to outbox ---\n{result.get('draft', '').strip()}")
    else:
        print("\n(no reply sent - escalated/rejected; see audit_log)")


if __name__ == "__main__":
    asyncio.run(main())