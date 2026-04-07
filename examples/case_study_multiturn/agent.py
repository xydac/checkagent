"""Multi-turn FAQ chatbot for the CheckAgent case study.

This agent handles FAQ queries across multiple conversation turns,
maintaining context from earlier turns to provide coherent follow-ups.
Demonstrates context preservation, tool sequencing, and state tracking.
"""

from __future__ import annotations

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall


async def faq_chatbot(
    input: AgentInput,
    *,
    llm: object,
    tools: object,
) -> AgentRun:
    """Handle an FAQ query with conversation context."""
    steps: list[Step] = []
    query = input.query
    history = input.conversation_history

    # Step 1: Understand query with conversation context
    context_prompt = query
    if history:
        recent = history[-4:]  # Last 2 exchanges
        context_prompt = (
            "Previous conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            + f"\n\nCurrent question: {query}"
        )

    response = await llm.complete(context_prompt)
    steps.append(Step(input_text=context_prompt, output_text=response))

    # Step 2: Search FAQ if this is a new topic (not a follow-up)
    is_followup = _is_followup(query, history)
    if not is_followup:
        faq_result = await tools.call("search_faq", {"query": query})
        steps.append(Step(
            input_text=f"search_faq({query})",
            output_text=str(faq_result),
            tool_calls=[
                ToolCall(name="search_faq", arguments={"query": query}, result=faq_result),
            ],
        ))

    # Step 3: Save preference if user expresses one
    if _expresses_preference(query):
        pref = _extract_preference(query)
        save_result = await tools.call("save_preference", {"preference": pref})
        steps.append(Step(
            input_text=f"save_preference({pref})",
            output_text=str(save_result),
            tool_calls=[
                ToolCall(
                    name="save_preference",
                    arguments={"preference": pref},
                    result=save_result,
                ),
            ],
        ))

    # Step 4: Generate final response incorporating context
    all_context = "\n".join(s.output_text for s in steps)
    final = await llm.complete(f"Answer based on: {all_context}")
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


def _is_followup(query: str, history: list[dict[str, str]]) -> bool:
    """Detect if query is a follow-up to previous conversation."""
    if not history:
        return False
    followup_signals = [
        "what about", "how about", "and also", "tell me more",
        "what else", "can you elaborate", "more detail",
        "you mentioned", "earlier you said", "going back to",
    ]
    lower = query.lower()
    return any(signal in lower for signal in followup_signals)


def _expresses_preference(query: str) -> bool:
    """Detect if the user expresses a preference."""
    pref_signals = ["i prefer", "i like", "i want", "please use", "switch to"]
    return any(s in query.lower() for s in pref_signals)


def _extract_preference(query: str) -> str:
    """Extract the preference value from the query."""
    lower = query.lower()
    for signal in ["i prefer", "i like", "i want", "please use", "switch to"]:
        if signal in lower:
            idx = lower.index(signal) + len(signal)
            return query[idx:].strip().rstrip(".")
    return query
