"""Sample customer service agent for replay testing demonstration.

This agent simulates a real LLM-backed agent.  In production it would call
OpenAI/Anthropic; here it uses deterministic logic so the example runs without
API keys.

Usage:
    cd examples/replay_cassette
    pytest tests/test_agent_replay.py -v       # first run: records cassettes
    pytest tests/test_agent_replay.py -v       # subsequent runs: replay only
"""
from __future__ import annotations

SCOPE = "order status, returns, and refunds"

_REFUSAL_KEYWORDS = [
    "ignore",
    "jailbreak",
    "system prompt",
    "override",
    "forget",
    "pretend",
    "act as",
]

_RETURN_POLICY = (
    "Our return policy allows returns within 30 days of purchase with a receipt. "
    "Items must be in original condition."
)

_REFUND_POLICY = "Refunds are processed within 5-7 business days after we receive the return."

_STATUS_TEMPLATE = "Order #{order_id} is currently {status}. Estimated delivery: {eta}."


async def customer_agent(prompt: str) -> str:
    """Customer service agent that answers order-related questions."""
    lower = prompt.lower()

    if any(kw in lower for kw in _REFUSAL_KEYWORDS):
        return (
            f"I'm sorry, I can only help with {SCOPE}. "
            "Please contact our team for other questions."
        )

    if "return" in lower and "policy" in lower:
        return _RETURN_POLICY

    if "refund" in lower:
        return _REFUND_POLICY

    if "order" in lower and any(c.isdigit() for c in prompt):
        import re

        order_id = re.search(r"\d+", prompt)
        oid = order_id.group() if order_id else "unknown"
        return _STATUS_TEMPLATE.format(
            order_id=oid,
            status="in transit",
            eta="2-3 business days",
        )

    return (
        f"I'm your customer service assistant. I can help with {SCOPE}. "
        "What can I assist you with?"
    )
