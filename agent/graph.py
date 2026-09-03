"""
M2: the triage graph, linear pass.

    START -> classify -> draft -> review -> send -> END

- classify : LLM structured output (category, priority, confidence)
- draft    : LLM with the two READ-ONLY tools bound; runs a bounded tool loop
             (search_kb + get_customer_history) then writes a grounded reply
- review   : LLM structured output (pass/fail + notes)
- send     : DETERMINISTIC - Python calls record_decision. In M2 every ticket
             is auto_sent; the real escalate-vs-send gate arrives in M3.

Content (classify/draft/review) is the model. Control (what happens at send)
is Python. That split is the whole design.
"""

from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

CATEGORIES = ("shipping_issue, shipping_status, returns_exchange, warranty, "
              "refund, billing, health_safety, general_question, other")


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
    final_action: str


def _text_of(content) -> str:
    """MCP tool / model content can be a list of blocks; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def build_graph(tools, model, max_tool_iters: int = 4):
    by_name = {t.name: t for t in tools}
    read_tools = [by_name["search_kb"], by_name["get_customer_history"]]

    classifier = model.with_structured_output(Classification)
    reviewer = model.with_structured_output(Review)
    drafter = model.bind_tools(read_tools)

    async def classify(state: TriageState):
        t = state["ticket"]
        msgs = [
            SystemMessage(content=(
                "You triage customer support tickets. Assign exactly one "
                f"category from: {CATEGORIES}. priority is 'high' for "
                "deadlines, urgency, legal or health/safety; else 'normal' or "
                "'low'. confidence is 0-1.")),
            HumanMessage(content=f"Subject: {t['subject']}\n\nBody: {t['body']}"),
        ]
        res = await classifier.ainvoke(msgs)
        return {"category": res.category, "priority": res.priority,
                "confidence": res.confidence}

    async def draft(state: TriageState):
        t = state["ticket"]
        msgs = [
            SystemMessage(content=(
                "You are a Harbor Goods support agent writing a reply to a "
                "customer. Before writing: call get_customer_history for the "
                "sender, and call search_kb for the relevant policy. Ground the "
                "reply strictly in what policy allows. Never promise a refund. "
                "When ready, write ONLY the final reply text with no tool call.")),
            HumanMessage(content=(
                f"From: {t['sender']}\n"
                f"Category: {state['category']} | Priority: {state['priority']}\n"
                f"Subject: {t['subject']}\nBody: {t['body']}")),
        ]
        resp = None
        for _ in range(max_tool_iters):
            resp = await drafter.ainvoke(msgs)
            msgs.append(resp)
            if not getattr(resp, "tool_calls", None):
                break
            for tc in resp.tool_calls:
                tool_msg = await by_name[tc["name"]].ainvoke(tc)
                msgs.append(tool_msg)
        return {"draft": _text_of(resp.content)}

    async def review(state: TriageState):
        msgs = [
            SystemMessage(content=(
                "You review a drafted support reply before it is sent. Return "
                "verdict='fail' if it promises a refund, invents policy, is "
                "off-topic, or commits to anything outside policy; else "
                "verdict='pass'.")),
            HumanMessage(content=(
                f"Category: {state['category']}\n\nDraft reply:\n{state['draft']}")),
        ]
        res = await reviewer.ainvoke(msgs)
        return {"review_verdict": res.verdict, "review_notes": res.notes}

    async def send(state: TriageState):
        # M2 is linear: everything is auto_sent. The gate is M3.
        t = state["ticket"]
        await by_name["record_decision"].ainvoke({
            "type": "tool_call", "name": "record_decision", "id": "send_1",
            "args": {
                "ticket_id": t["ticket_id"], "recipient": t["sender"],
                "action": "auto_sent", "body": state["draft"],
                "reason": f"M2 linear; review={state.get('review_verdict')}",
            },
        })
        return {"final_action": "auto_sent"}

    g = StateGraph(TriageState)
    g.add_node("classify", classify)
    g.add_node("draft", draft)
    g.add_node("review", review)
    g.add_node("send", send)
    g.add_edge(START, "classify")
    g.add_edge("classify", "draft")
    g.add_edge("draft", "review")
    g.add_edge("review", "send")
    g.add_edge("send", END)
    return g.compile()