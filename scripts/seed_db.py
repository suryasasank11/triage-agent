"""
M0 seed script for the triage agent.

Creates data/triage.db (SQLite) with four tables:
  - customers        : known customers (read by get_customer_history)
  - ticket_history   : prior resolved tickets per customer
  - outbox           : agent-written replies (starts EMPTY)
  - audit_log        : one row per decision the agent makes (starts EMPTY)

Safe to re-run: it drops and recreates the tables each time.
Uses only the Python standard library (sqlite3) -- no installs needed.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "triage.db"

CUSTOMERS = [
    # (email, name, signup_date, tier)
    ("maria.chen@example.com",      "Maria Chen",       "2023-04-11", "standard"),
    ("luis.romero@example.com",     "Luis Romero",      "2024-09-02", "standard"),
    ("sara.lindqvist@example.com",  "Sara Lindqvist",   "2025-06-20", "standard"),
    ("david.okoro@example.com",     "David Okoro",      "2022-01-15", "standard"),
    ("tom.becker@example.com",      "Tom Becker",       "2024-03-30", "standard"),
    ("amina.hassan@example.com",    "Amina Hassan",     "2021-11-08", "vip"),
]

# Prior resolved tickets. Some customers have history, some don't.
# Senders NOT listed here (greg.hall, priya.nair, kevin.brooks) are new -->
# get_customer_history should return nothing for them. That is intentional.
TICKET_HISTORY = [
    # (email, subject, category, resolved_date)
    ("maria.chen@example.com",   "Late delivery",        "shipping_status", "2024-12-02"),
    ("maria.chen@example.com",   "Exchange for larger",  "returns_exchange", "2025-02-18"),
    ("david.okoro@example.com",  "Damaged on arrival",   "warranty",        "2023-08-21"),
    ("tom.becker@example.com",   "Color exchange",       "returns_exchange", "2024-07-05"),
    ("amina.hassan@example.com", "Defective boots",      "warranty",        "2022-05-14"),
    ("amina.hassan@example.com", "Missing item",         "shipping_issue",  "2023-01-09"),
    ("amina.hassan@example.com", "Sizing advice",        "general_question", "2024-10-30"),
]


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clean slate on every run.
    for table in ("customers", "ticket_history", "outbox", "audit_log"):
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    cur.execute("""
        CREATE TABLE customers (
            email       TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            tier        TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE ticket_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT NOT NULL,
            subject       TEXT NOT NULL,
            category      TEXT NOT NULL,
            resolved_date TEXT NOT NULL,
            FOREIGN KEY (email) REFERENCES customers(email)
        )
    """)
    # Filled by the agent later -- start empty.
    cur.execute("""
        CREATE TABLE outbox (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id  TEXT NOT NULL,
            sender     TEXT NOT NULL,
            body       TEXT NOT NULL,
            action     TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur.execute("""
        CREATE TABLE audit_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id     TEXT NOT NULL,
            category      TEXT,
            priority      TEXT,
            confidence    REAL,
            review_verdict TEXT,
            final_action  TEXT,
            reason        TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cur.executemany(
        "INSERT INTO customers (email, name, signup_date, tier) VALUES (?, ?, ?, ?)",
        CUSTOMERS,
    )
    cur.executemany(
        "INSERT INTO ticket_history (email, subject, category, resolved_date) "
        "VALUES (?, ?, ?, ?)",
        TICKET_HISTORY,
    )

    conn.commit()
    conn.close()

    print(f"Seeded {DB_PATH}")
    print(f"  customers      : {len(CUSTOMERS)}")
    print(f"  ticket_history : {len(TICKET_HISTORY)}")
    print(f"  outbox         : 0 (agent writes here later)")
    print(f"  audit_log      : 0 (agent writes here later)")


if __name__ == "__main__":
    main()