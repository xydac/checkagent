"""Minimal RAG agent for the CheckAgent case study.

This agent answers questions by retrieving relevant documents from a
knowledge base and synthesizing answers. It demonstrates a realistic
retrieve-then-generate workflow in ~50 lines of logic.
"""

from __future__ import annotations

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall


async def rag_agent(
    query: str,
    *,
    llm: object,
    tools: object,
) -> AgentRun:
    """Answer a question by retrieving documents and synthesizing."""
    steps: list[Step] = []

    # Step 1: Retrieve relevant documents
    docs = await tools.call("retrieve_docs", {"query": query, "top_k": 3})
    steps.append(Step(
        input_text=query,
        output_text=f"Retrieved {len(docs)} documents",
        tool_calls=[
            ToolCall(
                name="retrieve_docs",
                arguments={"query": query, "top_k": 3},
                result=docs,
            ),
        ],
    ))

    # Step 2: Rank and filter documents by relevance
    ranked = await tools.call("rank_docs", {
        "query": query,
        "documents": docs,
    })
    steps.append(Step(
        input_text="rank_docs",
        output_text=f"Ranked to {len(ranked)} relevant docs",
        tool_calls=[
            ToolCall(
                name="rank_docs",
                arguments={"query": query, "documents": docs},
                result=ranked,
            ),
        ],
    ))

    # Step 3: Synthesize answer from ranked documents
    context = "\n---\n".join(
        f"[{d.get('title', 'untitled')}]: {d.get('content', d.get('snippet', ''))}"
        for d in ranked
    )
    prompt = (
        f"Based on the following documents, answer: {query}\n\n{context}"
    )
    answer = await llm.complete(prompt)
    steps.append(Step(input_text=prompt, output_text=answer))

    return AgentRun(
        input=AgentInput(query=query),
        steps=steps,
        final_output=answer,
    )
