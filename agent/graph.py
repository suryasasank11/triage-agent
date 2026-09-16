"""
M3: branching triage graph with human-in-the-loop.

                 classify
                    |
         (sensitive or conf<0.5)------> escalate <---------------.
                    | else                   ^                    |
                  draft <----.               | (fail at cap)      |
                    |        | (fail,         |                    |
                 review------' retries left)  |                    |
                    | pass                    |                    |
                  gate --(not auto-eligible)--'                    |
                    | eligible                                     |
                  send                                             |
                                                                   |
   escalate calls interrupt() -> pauses for a human -> approve/edit/reject

Bands on confidence:
  < 0.5           : skip drafting, escalate immediately
  0.5 <= c < 0.8  : draft + review, but NOT auto-send eligible -> human approves
  >= 0.8          : eligible to auto-send (if non-sensitive and review passed)

Content (classify/draft/review) = LLM. Control (routing/gate/escalate) = Python.
The final `action` field IS the contract the n8n layer will later route on.
"""

from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

CATEGORIES = ("shipping_issue, shipping_status, returns_exchange, warranty, "
              "refund, billing, health_safety, general_question, other")

SENSITIVE = {"refund", "billing", "health_safety"}
CLASSIFY_ESCALATE_CONF = 0.5   # below this: don't even draft
AUTO_SEND_CONF = 0.8           # below this: draft, but a human must approve
MAX_ATTEMPTS = 2               # draft attempts before giving up to a human


class Classification(BaseModel):
    category: str = Field(description=f"exactly one of: {CATEGORIES}")
    priority: Literal["low", "normal", "high"]
    confidence: float = Field(ge=0.0, le=1.0)


class Review(BaseModel):
    verdict: Literal["pass", "fail"]
    notes: str = Field(description="one sentence on why")


class TriageState(TypedDict, total=False):
    ticket: dict
    category: str
    priority: str
    confidence: float
    draft: str
    review_verdict: str
    review_notes: str
    attempts: int
    gate_decision: str
    escalation_reason: str
    human_decision: str
    final_action: str


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def build_graph(tools, model, checkpointer, max_tool_iters: int = 4):
    by_name = {t.name: t for t in tools}
    read_tools = [by_name["search_kb"], by_name["get_customer_history"]]

    classifier = model.with_structured_output(Classification)
    reviewer = model.with_structured_output(Review)
    drafter = model.bind_tools(read_tools)

    async def _record(ticket, action, body, reason):
        await by_name["record_decision"].ainvoke({
            "type": "tool_call", "name": "record_decision", "id": "rec",
            "args": {"ticket_id": ticket["ticket_id"], "recipient": ticket["sender"],
                     "action": action, "body": body, "reason": reason},
        })

    # ---------- nodes ----------

    async def classify(state):
        t = state["ticket"]
        res = await classifier.ainvoke([
            SystemMessage(content=(
                "You triage support tickets. Assign exactly one category from: "
                f"{CATEGORIES}. priority 'high' for deadlines/urgency/legal/health. "
                "confidence 0-1.")),
            HumanMessage(content=f"Subject: {t['subject']}\n\nBody: {t['body']}"),
        ])
        return {"category": res.category, "priority": res.priority,
                "confidence": res.confidence, "attempts": 0}

    async def draft(state):
        t = state["ticket"]
        retry_hint = ""
        if state.get("review_verdict") == "fail" and state.get("review_notes"):
            retry_hint = (f"\n\nYour previous draft was rejected: "
                          f"{state['review_notes']}. Fix that.")
        msgs = [
            SystemMessage(content=(
                "You are a Harbor Goods support agent writing a reply. Before "
                "writing, call get_customer_history for the sender and search_kb "
                "for the relevant policy. Ground the reply strictly in policy. "
                "Never promise a refund. Then write ONLY the final reply text."
                + retry_hint)),
            HumanMessage(content=(
                f"From: {t['sender']}\nCategory: {state['category']} | "
                f"Priority: {state['priority']}\nSubject: {t['subject']}\n"
                f"Body: {t['body']}")),
        ]
        resp = None
        for _ in range(max_tool_iters):
            resp = await drafter.ainvoke(msgs)
            msgs.append(resp)
            if not getattr(resp, "tool_calls", None):
                break
            for tc in resp.tool_calls:
                msgs.append(await by_name[tc["name"]].ainvoke(tc))
        return {"draft": _text_of(resp.content), "attempts": state.get("attempts", 0) + 1}

    async def review(state):
        res = await reviewer.ainvoke([
            SystemMessage(content=(
                "Review a drafted support reply. verdict='fail' if it promises a "
                "refund, invents policy, is off-topic, or commits outside policy; "
                "else 'pass'.")),
            HumanMessage(content=f"Category: {state['category']}\n\nDraft:\n{state['draft']}"),
        ])
        return {"review_verdict": res.verdict, "review_notes": res.notes}

    async def gate(state):
        # Deterministic auto-send eligibility. (Sensitive already escalated
        # earlier; checked again here as defense in depth.)
        if state["category"] in SENSITIVE:
            return {"gate_decision": "escalate",
                    "escalation_reason": "sensitive category"}
        if state["confidence"] < AUTO_SEND_CONF:
            return {"gate_decision": "escalate",
                    "escalation_reason": f"confidence {state['confidence']:.2f} below auto-send bar {AUTO_SEND_CONF}"}
        return {"gate_decision": "send"}

    async def send(state):
        await _record(state["ticket"], "auto_sent", state["draft"],
                      f"auto-sent; confidence={state['confidence']:.2f}")
        return {"final_action": "auto_sent"}

    async def escalate(state):
        # Derive the reason if an upstream node didn't set one (the
        # classify->escalate path routes without writing state).
        reason = state.get("escalation_reason")
        if not reason:
            if state["category"] in SENSITIVE:
                reason = f"sensitive category: {state['category']}"
            elif state["confidence"] < CLASSIFY_ESCALATE_CONF:
                reason = f"low confidence {state['confidence']:.2f}"
            else:
                reason = "policy requires human review"

        # PAUSE HERE for a human. On resume, `interrupt` returns their decision.
        # No side effects before interrupt(): the node re-runs from the top on
        # resume, so anything above interrupt() would execute twice.
        decision = interrupt({
            "ticket_id": state["ticket"]["ticket_id"],
            "reason": reason,
            "category": state.get("category"),
            "confidence": state.get("confidence"),
            "draft": state.get("draft", "(no draft - escalated before drafting)"),
        })
        # `decision` is what a human/n8n sends back: {"decision": ..., "body": ...}
        choice = (decision or {}).get("decision", "reject")
        if choice in ("approve", "edit"):
            body = (decision.get("body") or state.get("draft", "")).strip()
            await _record(state["ticket"], "auto_sent", body,
                          f"human {choice} after escalation: {reason}")
            return {"human_decision": choice, "draft": body,
                    "final_action": "human_approved"}
        await _record(state["ticket"], "escalated", "",
                      f"human rejected: {reason}")
        return {"human_decision": "reject", "final_action": "escalated_rejected"}

    # ---------- routers (conditional edges) ----------

    def after_classify(state) -> str:
        if state["category"] in SENSITIVE or state["confidence"] < CLASSIFY_ESCALATE_CONF:
            return "escalate"
        return "draft"

    def after_review(state) -> str:
        if state["review_verdict"] == "pass":
            return "gate"
        if state.get("attempts", 0) < MAX_ATTEMPTS:
            return "draft"
        return "escalate"

    def after_gate(state) -> str:
        return "send" if state["gate_decision"] == "send" else "escalate"

    # ---------- wiring ----------

    g = StateGraph(TriageState)
    for name, fn in [("classify", classify), ("draft", draft), ("review", review),
                     ("gate", gate), ("send", send), ("escalate", escalate)]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify, {"draft": "draft", "escalate": "escalate"})
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", after_review, {"gate": "gate", "draft": "draft", "escalate": "escalate"})
    g.add_conditional_edges("gate", after_gate, {"send": "send", "escalate": "escalate"})
    g.add_edge("send", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer)