"""
M0 verification. Reads back everything the scaffold created and prints a
report. If any check fails it prints [FAIL] and exits non-zero, so you know
M0 is not green yet.

Uses utf-8-sig when reading text files so a Windows UTF-8 BOM (which PowerShell
can add) is stripped cleanly and never breaks JSON parsing.
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB = DATA / "kb"
DB_PATH = DATA / "triage.db"
TICKETS = DATA / "tickets.jsonl"

ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    tag = "[ OK ]" if condition else "[FAIL]"
    if not condition:
        ok = False
    print(f"{tag} {label}" + (f"  ->  {detail}" if detail else ""))


print("=== KB files ===")
expected_kb = {"refund_policy.md", "shipping_policy.md",
               "returns_exchanges.md", "warranty.md"}
found_kb = {p.name for p in KB.glob("*.md")}
check("4 KB markdown files present", expected_kb <= found_kb,
      f"found {sorted(found_kb)}")

print("\n=== Ticket corpus ===")
tickets = []
with TICKETS.open(encoding="utf-8-sig") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
            tickets.append(t)
        except json.JSONDecodeError as e:
            check(f"line {i} is valid JSON", False, str(e))
check("12 tickets parsed", len(tickets) == 12, f"got {len(tickets)}")
required_keys = {"ticket_id", "sender", "subject", "body"}
all_keys_ok = all(required_keys <= set(t) for t in tickets)
check("every ticket has ticket_id/sender/subject/body", all_keys_ok)
ids = [t["ticket_id"] for t in tickets]
check("ticket_ids are unique", len(ids) == len(set(ids)))

print("\n=== SQLite ===")
check("triage.db exists", DB_PATH.exists())
if DB_PATH.exists():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    def count(tbl):
        return cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    check("customers seeded", count("customers") == 6, f"{count('customers')} rows")
    check("ticket_history seeded", count("ticket_history") == 7,
          f"{count('ticket_history')} rows")
    check("outbox starts empty", count("outbox") == 0)
    check("audit_log starts empty", count("audit_log") == 0)

    known = {r[0] for r in cur.execute("SELECT email FROM customers")}
    senders = {t["sender"] for t in tickets}
    new_senders = sorted(senders - known)
    print("\n=== Sender coverage (context for get_customer_history) ===")
    print(f"  known senders in corpus : {len(senders & known)}")
    print(f"  new senders in corpus   : {len(new_senders)}  -> {new_senders}")
    conn.close()

print("\n" + ("M0 GREEN -- all checks passed." if ok else "M0 NOT green -- fix the [FAIL]s above."))
sys.exit(0 if ok else 1)