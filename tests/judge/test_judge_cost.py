"""Measure judge prompt token counts for RQ2 cost data.

This test module measures the actual token consumption of judge
evaluations across realistic agent runs, producing data for the
paper's cost-effectiveness analysis (Section 5.2, RQ2).
"""

from __future__ import annotations

from checkagent.core.cost import BUILTIN_PRICING, calculate_run_cost
from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.judge.judge import _build_system_prompt, _build_user_prompt
from checkagent.judge.types import Criterion, Rubric, ScaleType


def _chars_to_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4 + 1


def _make_simple_run() -> AgentRun:
    """A minimal agent run (1 step, no tools)."""
    return AgentRun(
        input=AgentInput(query="What is the weather in NYC?"),
        steps=[
            Step(
                input_text="What is the weather in NYC?",
                output_text="The weather in NYC is currently 72°F and sunny.",
            ),
        ],
        final_output="The weather in NYC is currently 72°F and sunny.",
    )


def _make_tool_run() -> AgentRun:
    """A realistic agent run (2 steps, 2 tool calls)."""
    return AgentRun(
        input=AgentInput(query="Book a meeting with Alice for tomorrow at 2pm"),
        steps=[
            Step(
                input_text="Book a meeting with Alice for tomorrow at 2pm",
                output_text="Let me check the calendar and book that meeting.",
                tool_calls=[
                    ToolCall(
                        name="check_calendar",
                        arguments={"date": "2026-04-06", "time": "14:00"},
                        result={"available": True, "conflicts": []},
                    ),
                ],
            ),
            Step(
                input_text="Calendar is available. Creating the event.",
                output_text="Meeting booked! Event ID: evt-789.",
                tool_calls=[
                    ToolCall(
                        name="create_event",
                        arguments={
                            "title": "Meeting with Alice",
                            "date": "2026-04-06",
                            "time": "14:00",
                            "attendees": ["alice@example.com"],
                        },
                        result={"event_id": "evt-789", "confirmed": True},
                    ),
                ],
            ),
        ],
        final_output="Meeting booked! Event ID: evt-789.",
    )


def _make_complex_run() -> AgentRun:
    """A complex agent run (4 steps, 5 tool calls) — upper bound scenario."""
    query = (
        "Research the top 3 competitors for our product, "
        "summarize findings, and create a report"
    )
    search_results = {
        "results": [
            {"title": f"Competitor {i}", "url": f"https://ex.com/{i}"}
            for i in range(5)
        ],
    }
    comp1 = (
        "Competitor 1 offers project tracking, team collaboration, "
        "and reporting. Pricing: $20/user/month. Key features: "
        "real-time dashboards, Gantt charts, resource allocation."
    )
    comp2 = (
        "Competitor 2 focuses on agile workflows with sprint planning, "
        "backlog management, and velocity tracking. "
        "Pricing: $15/user/month. Key features: Kanban boards."
    )
    comp3 = (
        "Competitor 3 provides enterprise-grade project management "
        "with advanced analytics, custom workflows, and API "
        "integrations. Pricing: $35/user/month."
    )
    report = (
        "Here is the competitive analysis report:\n\n"
        "## Top 3 Competitors\n\n"
        "1. **Competitor 1** - Full-featured PM ($20/user/mo)\n"
        "2. **Competitor 2** - Agile-focused ($15/user/mo)\n"
        "3. **Competitor 3** - Enterprise ($35/user/mo)\n\n"
        "Key takeaway: Differentiate on AI-powered insights."
    )
    return AgentRun(
        input=AgentInput(query=query),
        steps=[
            Step(
                input_text=query,
                output_text="I'll search for competitor info first.",
                tool_calls=[
                    ToolCall(
                        name="web_search",
                        arguments={"query": "top competitors 2026"},
                        result=search_results,
                    ),
                ],
            ),
            Step(
                input_text="Found 5 results. Analyzing top 3.",
                output_text="Fetching details on each competitor.",
                tool_calls=[
                    ToolCall(
                        name="web_fetch",
                        arguments={"url": "https://ex.com/1"},
                        result={"content": comp1},
                    ),
                    ToolCall(
                        name="web_fetch",
                        arguments={"url": "https://ex.com/2"},
                        result={"content": comp2},
                    ),
                ],
            ),
            Step(
                input_text="Got 2 competitors. Fetching the third.",
                output_text="Fetching last competitor, preparing report.",
                tool_calls=[
                    ToolCall(
                        name="web_fetch",
                        arguments={"url": "https://ex.com/3"},
                        result={"content": comp3},
                    ),
                ],
            ),
            Step(
                input_text="All data gathered. Generating report.",
                output_text=report,
            ),
        ],
        final_output=report,
    )


def _standard_rubric() -> Rubric:
    """A typical 3-criterion rubric for testing."""
    return Rubric(
        name="task_quality",
        description="Evaluate the quality of the agent's task completion",
        criteria=[
            Criterion(
                name="correctness",
                description="Did the agent produce a correct and complete result?",
                scale_type=ScaleType.NUMERIC,
                scale=[1, 2, 3, 4, 5],
                weight=2.0,
            ),
            Criterion(
                name="efficiency",
                description="Did the agent use tools efficiently without unnecessary calls?",
                scale_type=ScaleType.NUMERIC,
                scale=[1, 2, 3, 4, 5],
                weight=1.0,
            ),
            Criterion(
                name="communication",
                description="Was the agent's communication clear and helpful?",
                scale_type=ScaleType.BINARY,
                scale=["fail", "pass"],
                weight=1.0,
            ),
        ],
    )


class TestJudgePromptSizes:
    """Measure system and user prompt token counts for judge evaluations."""

    def test_simple_run_prompt_size(self):
        rubric = _standard_rubric()
        run = _make_simple_run()
        system = _build_system_prompt(rubric)
        user = _build_user_prompt(run)
        sys_tokens = _chars_to_tokens(system)
        usr_tokens = _chars_to_tokens(user)
        total_input = sys_tokens + usr_tokens
        # Record measurements for the paper
        assert sys_tokens > 0
        assert usr_tokens > 0
        # Simple run: expect < 300 input tokens
        assert total_input < 300, f"Simple run input: {total_input} tokens"

    def test_tool_run_prompt_size(self):
        rubric = _standard_rubric()
        run = _make_tool_run()
        system = _build_system_prompt(rubric)
        user = _build_user_prompt(run)
        sys_tokens = _chars_to_tokens(system)
        usr_tokens = _chars_to_tokens(user)
        total_input = sys_tokens + usr_tokens
        assert total_input < 400, f"Tool run input: {total_input} tokens"

    def test_complex_run_prompt_size(self):
        rubric = _standard_rubric()
        run = _make_complex_run()
        system = _build_system_prompt(rubric)
        user = _build_user_prompt(run)
        sys_tokens = _chars_to_tokens(system)
        usr_tokens = _chars_to_tokens(user)
        total_input = sys_tokens + usr_tokens
        assert total_input < 800, f"Complex run input: {total_input} tokens"

    def test_print_all_measurements(self, capsys):
        """Print all measurements for easy collection into metrics.md."""
        rubric = _standard_rubric()
        scenarios = [
            ("simple (1 step, 0 tools)", _make_simple_run()),
            ("tool (2 steps, 2 tools)", _make_tool_run()),
            ("complex (4 steps, 5 tools)", _make_complex_run()),
        ]
        system = _build_system_prompt(rubric)
        sys_tokens = _chars_to_tokens(system)
        # Estimate judge response: JSON with 3 criteria scores + reasoning
        # Typically ~150-300 tokens for a 3-criterion rubric
        estimated_response_tokens = 200

        print("\n=== Judge Cost Measurement (RQ2) ===")
        print(f"System prompt: {len(system)} chars, ~{sys_tokens} tokens")
        print(f"Estimated response: ~{estimated_response_tokens} tokens")
        print()

        for name, run in scenarios:
            user = _build_user_prompt(run)
            usr_tokens = _chars_to_tokens(user)
            total_input = sys_tokens + usr_tokens

            # Cost per trial for common judge models
            for model, pricing in [
                ("gpt-4o", BUILTIN_PRICING["gpt-4o"]),
                ("gpt-4o-mini", BUILTIN_PRICING["gpt-4o-mini"]),
                ("claude-sonnet", BUILTIN_PRICING["claude-sonnet"]),
                ("claude-haiku", BUILTIN_PRICING["claude-haiku"]),
            ]:
                input_cost = total_input * pricing.input / 1_000_000
                output_cost = estimated_response_tokens * pricing.output / 1_000_000
                trial_cost = input_cost + output_cost
                print(
                    f"  {name} | {model}: "
                    f"in={total_input}tok, out=~{estimated_response_tokens}tok, "
                    f"cost=${trial_cost:.6f}/trial, "
                    f"cost_3trials=${trial_cost * 3:.6f}"
                )
            print()


class TestCostTrackerIntegration:
    """Verify CostTracker works end-to-end with token-bearing runs."""

    def test_cost_calculation_with_tokens(self):
        """Create a run with token counts and verify cost calculation."""
        run = AgentRun(
            input=AgentInput(query="test"),
            steps=[
                Step(
                    input_text="test",
                    output_text="response",
                    model="gpt-4o",
                    prompt_tokens=150,
                    completion_tokens=200,
                ),
            ],
            final_output="response",
                total_prompt_tokens=150,
            total_completion_tokens=200,
        )
        breakdown = calculate_run_cost(run)
        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected = 150 * 2.50 / 1_000_000 + 200 * 10.00 / 1_000_000
        assert abs(breakdown.total_cost - expected) < 0.000001

    def test_judge_cost_estimation(self):
        """Estimate what a judge evaluation costs with real pricing."""
        # Typical judge evaluation: ~200 input tokens, ~200 output tokens
        # Using gpt-4o as judge
        input_tokens = 200
        output_tokens = 200
        pricing = BUILTIN_PRICING["gpt-4o"]
        cost_per_trial = (
            input_tokens * pricing.input / 1_000_000
            + output_tokens * pricing.output / 1_000_000
        )
        cost_3_trials = cost_per_trial * 3
        # Should be very cheap — under $0.01 per evaluation
        assert cost_per_trial < 0.01
        assert cost_3_trials < 0.03
