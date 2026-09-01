# Harbor Triage Agent

An agentic support-ticket triage system. It reads an incoming support ticket,
classifies it, drafts a policy-grounded reply, reviews that draft against the
rules, and then **either auto-sends it or escalates it to a human** — where the
send-vs-escalate decision is made by deterministic graph logic, not the model.

> Status: **in progress.** Data layer and MCP tool server complete (M0–M1);
> LangGraph agent, branching + human-in-the-loop, eval layer, and dashboard to
> follow (M2–M6).

## Why it's built this way

The agent decides *what to say*; the graph decides *what's allowed to happen*.
Content (classify, draft, review) is the LLM. Control (retry cap, auto-send
eligibility, forced escalation for sensitive categories) is plain Python in the
graph. Two independent safety layers: sensitive categories (refunds, billing,
health/legal) are force-escalated regardless of model confidence, and replies
are written to a local outbox — nothing is emailed.

## Architecture

    incoming ticket
          │
          ▼
      classify ──(urgent/sensitive)──► escalate ──► human review
          │
       draft ◄────┐
          │       │ (review failed, retries left)
       review ────┘
          │ pass
        gate ──(not auto-send eligible)──► escalate
          │ eligible
        send ──► outbox

Tools (`search_kb`, `get_customer_history`, `record_decision`) are exposed over
a **Model Context Protocol (MCP)** server and loaded into the graph via
`langchain-mcp-adapters`.

## Stack

- LangGraph (agent graph)
- MCP Python SDK (self-written tool server, stdio transport)
- SQLite + markdown (data layer, no external services)
- Anthropic API (the model)

## Run it

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

    python scripts\seed_db.py      # build the SQLite data layer
    python scripts\verify.py       # verify the scaffold (M0)
    python scripts\test_tools.py   # round-trip test the MCP server (M1)

## Layout

    data/           KB markdown, ticket corpus, seeded SQLite
    mcp_server/     the MCP tool server
    scripts/        seed + verification/test scripts