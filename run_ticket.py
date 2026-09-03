"""
M2 runner: load the MCP tools, plug in the real Anthropic model, and run ONE
ticket end to end through the linear graph.

Usage (PowerShell):
    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # or put it in a .env file
    python run_ticket.py T-1001
Defaults to T-1001 if no id is given.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.graph import build_graph

load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file if present

ROOT = Path(__file__).resolve().parent
SERVER = str(ROOT / "mcp_server" / "server.py")
TICKETS = ROOT / "data" / "tickets.jsonl"

# Fast + cheap for dev/iteration. If classification or review quality is weak,
# bump to "claude-sonnet-5" - it's a one-line swap.
MODEL = "claude-haiku-4-5-20251001"


def load_ticket(ticket_id: str) -> dict:
    with TICKETS.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["ticket_id"] == ticket_id:
                return t
    raise SystemExit(f"Ticket {ticket_id} not found in {TICKETS.name}")


async def main() -> None:
    ticket_id = sys.argv[1] if len(sys.argv) > 1 else "T-1001"
    ticket = load_ticket(ticket_id)

    client = MultiServerMCPClient({
        "harbor": {"command": sys.executable, "args": [SERVER], "transport": "stdio"}
    })
    tools = await client.get_tools()
    model = ChatAnthropic(model=MODEL, temperature=0, max_tokens=1024)
    graph = build_graph(tools, model)

    print(f"\n>>> {ticket_id}  |  {ticket['subject']}")
    print(f"    from {ticket['sender']}\n")

    final = await graph.ainvoke({"ticket": ticket})

    print(f"category   : {final.get('category')}  "
          f"(priority={final.get('priority')}, confidence={final.get('confidence')})")
    print(f"review     : {final.get('review_verdict')} - {final.get('review_notes')}")
    print(f"action     : {final.get('final_action')}")
    print("\n--- drafted reply ---")
    print(final.get("draft", "").strip())
    print("\n(recorded to outbox + audit_log; inspect with the query below)")


if __name__ == "__main__":
    asyncio.run(main())