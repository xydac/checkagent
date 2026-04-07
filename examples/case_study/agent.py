"""Minimal customer support agent for the CheckAgent case study.

This agent handles order inquiries by looking up order details and
checking refund eligibility. It demonstrates a realistic tool-using
workflow in ~30 lines of logic.
"""

from __future__ import annotations

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall


async def support_agent(
    query: str,
    *,
    llm: object,
    tools: object,
) -> AgentRun:
    """Handle a customer support query about an order."""
    steps: list[Step] = []

    # Step 1: Understand the request
    response = await llm.complete(query)
    steps.append(Step(input_text=query, output_text=response))

    # Step 2: Look up order details
    order_id = _extract_order_id(query)
    if order_id:
        order = await tools.call("lookup_order", {"order_id": order_id})
        steps.append(Step(
            input_text=f"lookup_order({order_id})",
            output_text=str(order),
            tool_calls=[
                ToolCall(name="lookup_order", arguments={"order_id": order_id}, result=order),
            ],
        ))

        # Step 3: Check refund policy if customer asks about refund
        if "refund" in query.lower() or "return" in query.lower():
            policy = await tools.call("check_refund_policy", {"order_id": order_id})
            steps.append(Step(
                input_text=f"check_refund_policy({order_id})",
                output_text=str(policy),
                tool_calls=[
                    ToolCall(
                        name="check_refund_policy",
                        arguments={"order_id": order_id},
                        result=policy,
                    ),
                ],
            ))

    # Step 4: Generate final response
    context = "\n".join(s.output_text for s in steps)
    final = await llm.complete(f"Summarize for customer: {context}")
    steps.append(Step(input_text="summarize", output_text=final))

    return AgentRun(input=AgentInput(query=query), steps=steps, final_output=final)


def _extract_order_id(text: str) -> str | None:
    """Extract order ID like ORD-123 from text."""
    import re
    match = re.search(r"ORD-\d+", text)
    return match.group(0) if match else None
