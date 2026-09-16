"""
M4: offline eval over the gold set.

For each gold ticket we run the graph and read the AGENT's decision:
  - if the graph PAUSES (hits __interrupt__), the agent chose to escalate
  - otherwise it auto-sent
We compare that against the gold `should_escalate` label and the gold category.

Metrics:
  - Classification accuracy (+ a list of misclassifications)
  - Escalation precision / recall
  - FALSE AUTO-SEND count  <-- the safety-critical error, tracked on its own
    (agent auto-sent something a human should have handled)
  - Over-escalation count  (safe but annoying: escalated something routine)

Writes a per-ticket trace to eval/last_run_trace.jsonl.

Run (PowerShell):  python -m eval.run_eval
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
SERVER = str(ROOT / "mcp_server" / "server.py")
TICKETS = ROOT / "data" / "tickets.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
TRACE = ROOT / "eval" / "last_run_trace.jsonl"
DB = ROOT / "data" / "triage.db"
MODEL = "claude-haiku-4-5-20251001"


def load_jsonl(p: Path):
    with p.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate(tools, model):
    tickets = {t["ticket_id"]: t for t in load_jsonl(TICKETS)}
    rows = []
    for g in load_jsonl(GOLD):
        tid = g["ticket_id"]
        graph = build_graph(tools, model, InMemorySaver())
        cfg = {"configurable": {"thread_id": f"eval-{tid}"}}
        result = await graph.ainvoke({"ticket": tickets[tid]}, cfg)

        pred_escalate = "__interrupt__" in result
        pred_cat = result.get("category")
        gold_escalate = g["should_escalate"]

        if gold_escalate and pred_escalate:
            outcome = "TP"           # correctly escalated
        elif not gold_escalate and not pred_escalate:
            outcome = "TN"           # correctly auto-sent
        elif not gold_escalate and pred_escalate:
            outcome = "FP"           # over-escalation (safe, annoying)
        else:
            outcome = "FN"           # FALSE AUTO-SEND (dangerous)

        rows.append({
            "ticket_id": tid,
            "gold_category": g["expected_category"],
            "pred_category": pred_cat,
            "category_correct": pred_cat == g["expected_category"],
            "gold_escalate": gold_escalate,
            "pred_escalate": pred_escalate,
            "confidence": result.get("confidence"),
            "outcome": outcome,
        })
    return rows


def summarize(rows) -> dict:
    n = len(rows)
    cat_ok = sum(r["category_correct"] for r in rows)
    tp = sum(r["outcome"] == "TP" for r in rows)
    tn = sum(r["outcome"] == "TN" for r in rows)
    fp = sum(r["outcome"] == "FP" for r in rows)
    fn = sum(r["outcome"] == "FN" for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {"n": n, "cat_acc": cat_ok / n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec}


def _fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def print_report(rows, s):
    print("\n" + "=" * 64)
    print("  M4 EVAL REPORT")
    print("=" * 64)
    print(f"  tickets evaluated       : {s['n']}")
    print(f"  classification accuracy : {s['cat_acc']:.0%}  "
          f"({sum(r['category_correct'] for r in rows)}/{s['n']})")
    print("\n  --- escalation decision ---")
    print(f"  precision               : {_fmt(s['precision'])}")
    print(f"  recall                  : {_fmt(s['recall'])}")
    print(f"  correct escalate  (TP)  : {s['tp']}")
    print(f"  correct auto-send (TN)  : {s['tn']}")
    print(f"  over-escalation   (FP)  : {s['fp']}   (safe but annoying)")
    print(f"  FALSE AUTO-SEND   (FN)  : {s['fn']}   <-- safety-critical")
    misclass = [r for r in rows if not r["category_correct"]]
    if misclass:
        print("\n  --- misclassifications ---")
        for r in misclass:
            print(f"    {r['ticket_id']}: gold={r['gold_category']} "
                  f"pred={r['pred_category']}")
    false_sends = [r for r in rows if r["outcome"] == "FN"]
    if false_sends:
        print("\n  --- FALSE AUTO-SENDS (investigate these) ---")
        for r in false_sends:
            print(f"    {r['ticket_id']}: auto-sent, should have escalated "
                  f"(pred_cat={r['pred_category']}, conf={r['confidence']})")
    print("=" * 64)


def cleanup_eval_rows(gold_ids):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    qs = ",".join("?" * len(gold_ids))
    cur.execute(f"DELETE FROM outbox WHERE ticket_id IN ({qs})", gold_ids)
    cur.execute(f"DELETE FROM audit_log WHERE ticket_id IN ({qs})", gold_ids)
    conn.commit()
    conn.close()


async def main():
    client = MultiServerMCPClient({
        "harbor": {"command": sys.executable, "args": [SERVER], "transport": "stdio"}})
    tools = await client.get_tools()
    model = ChatAnthropic(model=MODEL, temperature=0, max_tokens=1024)

    print("Running eval over the gold set (one graph run per ticket)...")
    rows = await evaluate(tools, model)

    TRACE.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = summarize(rows)
    print_report(rows, s)
    print(f"\nPer-ticket trace written to {TRACE.relative_to(ROOT)}")

    cleanup_eval_rows([r["ticket_id"] for r in rows])


if __name__ == "__main__":
    asyncio.run(main())