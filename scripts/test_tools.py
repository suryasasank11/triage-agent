import asyncio
import sqlite3
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parent.parent
SERVER = ROOT /"mcp_server"/"server.py"
DB_PATH = ROOT/"data"/"triage.db"

def _text(result):
    return '\n'.join(b.text for b in result.content if getattr(b,"type",None)=="text")

async def main()->int:
    params=StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()

            tools=await session.list_tools()
            names=[t.name for t in tools.tools]
            print("=== Tools exposed by the server ===")
            print(" ", names)
            expected = {"search_kb", "get_customer_history", "record_decision"}
            assert expected <= set(names),f"missing_tools: {expected - set(names)}"

            print("\n=== search_kb('wrong size exchange before my trip') ===")
            r=await session.call_tool("search_kb", {"query": "wrong size exchange before my trip deadline"})
            print(_text(r))

            print("\n=== get_customer_history(known : amina.hassan) ===")
            r=await session.call_tool("get_customer_history", {"email":"amina.hassan@example.com"})
            print(_text(r))

            print("\n=== get_customer_history (NEW: greg.hall) ===")
            r = await session.call_tool("get_customer_history", {"email": "greg.hall@example.com"})
            print(_text(r))

            print("\n=== record_decision (auto_sent smoke test) ===")
            r = await session.call_tool("record_decision", {
                "ticket_id": "T-TEST",
                "recipient": "maria.chen@example.com",
                "action": "auto_sent",
                "body": "Test reply body.",
                "reason": "m1 smoke test",
            })
            print(_text(r))


    conn = sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    ob=cur.execute("SELECT COUNT(*) FROM outbox WHERE ticket_id='T-TEST'").fetchone()[0]
    al = cur.execute("SELECT COUNT(*) from audit_log WHERE ticket_id='T-TEST'").fetchone()[0]
    print(f"\n=== Side-effect check ===\n  outbox rows for T-TEST   : {ob}\n  audit_log rows for T-TEST: {al}")
    cur.execute("DELETE FROM outbox WHERE ticket_id='T-TEST'")
    cur.execute("DELETE FROM audit_log WHERE ticket_id='T-TEST'")
    conn.commit()
    conn.close()
    print("  (test rows cleaned up - DB back to seeded state)")

    ok=ob==1 and al==1
    print("\n" + ("M1 GREEN -- server speaks MCP and all three tools work." if ok
                    else "M1 NOT green -- side effect did not land."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))