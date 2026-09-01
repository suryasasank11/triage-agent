import re
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT= Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB_DIR = DATA / "kb"
DB_PATH = DATA / "triage.db"

mcp = FastMCP("harvor-triage")

def _kb_chunks():
    """Split every KB markdown file into paragraph_sized chunks."""
    chunks=[]
    for path in sorted(KB_DIR.glob("*.md")):
        text=path.read_text(encoding="utf-8-sig")
        for para in (p.strip() for p in text.split("\n\n")):
            if para:
                chunks.append((path.stem,para))
    return chunks

@mcp.tool()
def search_kb(query:str)->str:
    """ Here we have the query and it is made into words and where that word
    are compared again with each of the documents text in the knowledge base and then it
    calclulates scores where it is no. of words match for each document, and i returns a list 
    of scores for each document"""
    q_terms=set(re.findall(r"[a-z0-9]+",query.lower()))
    scored=[]
    for source,text in _kb_chunks():
        t_terms=re.findall(r"[a-z0-9]+", text.lower())
        overlap=sum(1 for w in t_terms if w in q_terms)
        if overlap:
            scored.append((overlap,source,text))
    scored.sort(key=lambda x: -x[0])
    top=scored[:3]
    if not top:
        return "No relevant policy found for that query."
    return "\n\n\n\n".join(f"[source:{s}]\n{txt}" for _,s,txt in top)

@mcp.tool()
def get_customer_history(email: str)->str:
    """Here we fist connect to the database and then as the input is email we first search if there are any customers with this email
    If there are none return no customer and treat them as new cutomers with no history
    If customer exists then from the row then fetch (name,signup,tier) into 3 variables
    then you run a sql command to get the all the history of the cutomer from ticket_history table
    then print all the history of each customer with an (email)
    """

    conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    row=cur.execute(
        "SELECT name,signup_date,tier FROM customers WHERE email=?",(email,),
        ).fetchone()
    if row is None:
        conn.close()
        return f"No customer record found for {email}. Treat as a New customer with no history."
    name,signup,tier=row
    hist=cur.execute(
        "SELECT subject, category,resolved_date FROM ticket_history "
        "WHERE email=? ORDER BY resolved_date",
        (email,),
    ).fetchall()
    conn.close()
    lines=[f"Customer: {name} <{email}> | tier={tier} | since {signup}"]
    if hist:
        lines.append(f"{len(hist)} prior ticket(s):")
        lines+=[f"  - {d}: {subj} [{cat}]" for subj,cat,d in hist]
    else:
        lines.append("No prior tickets on record.")
    return "\n".join(lines)


@mcp.tool()
def record_decision(
    ticket_id:str,
    recipient:str,
    action:str,
    body:str="",
    reason:str="",
)->str:
    """Record the final outcome of handling a ticket. ALWAYS call this once per
    ticket as the last step. `action` is 'auto_sent' (an approved reply goes to
    the customer) or 'escalated' (handed to a human, no reply sent). When
    action is 'auto_sent', `body` is the reply text and it is written to the
    outbox. `reason` is a short justification. Nothing is emailed."""
    conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    cur.execute("INSERT INTO audit_log (ticket_id,final_action,reason) VALUES (?,?,?)",
                (ticket_id,action,reason),
                )
    wrote=False
    if action=="auto_sent":
        cur.execute(
            "INSERT INTO outbox (ticket_id,sender,body,action) VALUES (?,?,?,?)",
            (ticket_id,recipient,body,action),
        )
        wrote=True
    conn.commit()
    conn.close()
    msg=f"Recorded '{action}' for{ticket_id}."
    if wrote:
        msg+="Reply written to outbox."
    return msg


if __name__=="__main__":
    mcp.run(transport="stdio")

