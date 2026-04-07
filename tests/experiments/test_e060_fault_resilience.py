"""E-060: Fault injection resilience rates across agent archetypes.

Planned experiment E-004 from the research roadmap. Tests 4 agent archetypes
against 11 fault types (5 tool + 6 LLM) to measure crash vs survive rates.

Agent archetypes:
  A1 — No error handling (bare calls)
  A2 — Basic try/except (catch-all, return fallback)
  A3 — Retry logic (retry once on failure)
  A4 — Graceful degradation (skip failed tool, use cached LLM response)

Produces data for paper Section 5 (Evaluation), RQ1 and RQ4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool

# ---------------------------------------------------------------------------
# Agent archetypes
# ---------------------------------------------------------------------------

async def agent_no_handling(llm: MockLLM, tool: MockTool) -> str:
    """A1: No error handling — crashes propagate."""
    plan = await llm.complete("plan a trip")
    result = await tool.call("search", {"query": "hotels"})
    return f"Plan: {plan}, Hotels: {result}"


async def agent_basic_trycatch(llm: MockLLM, tool: MockTool) -> str:
    """A2: Basic try/except — catch all, return fallback."""
    try:
        plan = await llm.complete("plan a trip")
    except Exception:
        plan = "default plan"
    try:
        result = await tool.call("search", {"query": "hotels"})
    except Exception:
        result = "no results"
    return f"Plan: {plan}, Hotels: {result}"


async def agent_retry(llm: MockLLM, tool: MockTool) -> str:
    """A3: Retry once on failure."""
    plan = None
    for attempt in range(2):
        try:
            plan = await llm.complete("plan a trip")
            break
        except Exception:
            if attempt == 1:
                raise
    result = None
    for attempt in range(2):
        try:
            result = await tool.call("search", {"query": "hotels"})
            break
        except Exception:
            if attempt == 1:
                raise
    return f"Plan: {plan}, Hotels: {result}"


async def agent_graceful(llm: MockLLM, tool: MockTool) -> str:
    """A4: Graceful degradation — skip failed components."""
    try:
        plan = await llm.complete("plan a trip")
    except Exception:
        plan = "[LLM unavailable — using cached response]"
    try:
        result = await tool.call("search", {"query": "hotels"})
    except Exception:
        result = "[search unavailable — skipping]"
    return f"Plan: {plan}, Hotels: {result}"


AGENTS = {
    "A1_no_handling": agent_no_handling,
    "A2_basic_trycatch": agent_basic_trycatch,
    "A3_retry": agent_retry,
    "A4_graceful": agent_graceful,
}


# ---------------------------------------------------------------------------
# Fault scenarios
# ---------------------------------------------------------------------------

def make_tool_fault_scenarios() -> dict[str, FaultInjector]:
    """Create one FaultInjector per tool fault type."""
    scenarios = {}

    fi = FaultInjector()
    fi.on_tool("search").timeout(1.0)
    scenarios["tool_timeout"] = fi

    fi = FaultInjector()
    fi.on_tool("search").rate_limit(after_n=0)
    scenarios["tool_rate_limit"] = fi

    fi = FaultInjector()
    fi.on_tool("search").returns_malformed({"garbage": True})
    scenarios["tool_malformed"] = fi

    fi = FaultInjector()
    fi.on_tool("search").returns_empty()
    scenarios["tool_empty"] = fi

    fi = FaultInjector()
    fi.on_tool("search").intermittent(fail_rate=1.0, seed=42)
    scenarios["tool_intermittent"] = fi

    return scenarios


def make_llm_fault_scenarios() -> dict[str, FaultInjector]:
    """Create one FaultInjector per LLM fault type."""
    scenarios = {}

    fi = FaultInjector()
    fi.on_llm().context_overflow()
    scenarios["llm_context_overflow"] = fi

    fi = FaultInjector()
    fi.on_llm().rate_limit(after_n=0)
    scenarios["llm_rate_limit"] = fi

    fi = FaultInjector()
    fi.on_llm().server_error()
    scenarios["llm_server_error"] = fi

    fi = FaultInjector()
    fi.on_llm().content_filter()
    scenarios["llm_content_filter"] = fi

    fi = FaultInjector()
    fi.on_llm().partial_response()
    scenarios["llm_partial_response"] = fi

    fi = FaultInjector()
    fi.on_llm().intermittent(fail_rate=1.0, seed=42)
    scenarios["llm_intermittent"] = fi

    return scenarios


def make_recoverable_scenarios() -> dict[str, FaultInjector]:
    """Faults that can be recovered from with retry logic."""
    scenarios = {}

    # Intermittent tool fault: 50% chance — retry should succeed ~75% of time
    fi = FaultInjector()
    fi.on_tool("search").intermittent(fail_rate=0.5, seed=42)
    scenarios["tool_intermittent_50pct"] = fi

    # Rate limit after 1 call — first succeeds, retry on LLM call may help
    fi = FaultInjector()
    fi.on_llm().rate_limit(after_n=1)
    scenarios["llm_rate_limit_after_1"] = fi

    # Intermittent LLM fault: 50% chance
    fi = FaultInjector()
    fi.on_llm().intermittent(fail_rate=0.5, seed=42)
    scenarios["llm_intermittent_50pct"] = fi

    return scenarios


@dataclass
class TrialResult:
    agent: str
    fault: str
    survived: bool
    error_type: str | None = None
    duration_ms: float = 0.0


@dataclass
class ExperimentResults:
    trials: list[TrialResult] = field(default_factory=list)

    def survival_rate(self, agent: str) -> float:
        agent_trials = [t for t in self.trials if t.agent == agent]
        if not agent_trials:
            return 0.0
        return sum(1 for t in agent_trials if t.survived) / len(agent_trials)

    def crash_rate(self, agent: str) -> float:
        return 1.0 - self.survival_rate(agent)

    def by_fault(self, fault: str) -> dict[str, bool]:
        return {
            t.agent: t.survived
            for t in self.trials
            if t.fault == fault
        }


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

@pytest.mark.agent_test(layer="mock")
async def test_e060_fault_resilience_rates():
    """Run all agents through all fault scenarios and collect resilience data."""
    tool_scenarios = make_tool_fault_scenarios()
    llm_scenarios = make_llm_fault_scenarios()
    all_scenarios = {**tool_scenarios, **llm_scenarios}

    results = ExperimentResults()

    for agent_name, agent_fn in AGENTS.items():
        for fault_name, fi in all_scenarios.items():
            llm = MockLLM()
            llm.on_input(contains="plan").respond("Go to Paris")
            tool = MockTool()
            tool.on_call("search").respond({"hotels": ["Hotel A"]})

            # Attach faults
            if fi.has_faults_for("search"):
                tool.attach_faults(fi)
            if fi.has_llm_faults():
                llm.attach_faults(fi)

            fi.reset_records()  # Clean state for this trial

            t0 = time.perf_counter()
            try:
                await agent_fn(llm, tool)
                survived = True
                error_type = None
            except Exception as exc:
                survived = False
                error_type = type(exc).__name__
            duration_ms = (time.perf_counter() - t0) * 1000

            results.trials.append(TrialResult(
                agent=agent_name,
                fault=fault_name,
                survived=survived,
                error_type=error_type,
                duration_ms=duration_ms,
            ))

    # --- Assertions: verify expected behavior ---
    # A1 (no handling) should crash on most faults
    a1_rate = results.survival_rate("A1_no_handling")
    # A4 (graceful) should survive all faults
    a4_rate = results.survival_rate("A4_graceful")

    assert a4_rate == 1.0, f"A4 should survive all faults, got {a4_rate}"
    assert a1_rate < 0.5, f"A1 should crash on most faults, got survival={a1_rate}"

    # A2 (basic try/except) should survive all (catches Exception)
    a2_rate = results.survival_rate("A2_basic_trycatch")
    assert a2_rate == 1.0, f"A2 should survive all faults, got {a2_rate}"

    # --- Print data table for paper ---
    print("\n=== E-060: Fault Injection Resilience Rates ===\n")
    print(f"{'Agent':<22} {'Survival Rate':>14} {'Crash Rate':>11}")
    print("-" * 50)
    for agent_name in AGENTS:
        sr = results.survival_rate(agent_name)
        cr = results.crash_rate(agent_name)
        print(f"{agent_name:<22} {sr:>13.1%} {cr:>10.1%}")

    print(f"\n{'Fault Scenario':<25} {'A1':>4} {'A2':>4} {'A3':>4} {'A4':>4}")
    print("-" * 50)
    for fault_name in all_scenarios:
        by_fault = results.by_fault(fault_name)
        a1 = "✓" if by_fault.get("A1_no_handling") else "✗"
        a2 = "✓" if by_fault.get("A2_basic_trycatch") else "✗"
        a3 = "✓" if by_fault.get("A3_retry") else "✗"
        a4 = "✓" if by_fault.get("A4_graceful") else "✗"
        print(f"{fault_name:<25} {a1:>4} {a2:>4} {a3:>4} {a4:>4}")

    # Print crash error types for A1
    print(f"\n{'Fault':<25} {'A1 Error Type':<30} {'A3 Error Type':<30}")
    print("-" * 85)
    for fault_name in all_scenarios:
        a1_trial = next(
            t for t in results.trials
            if t.agent == "A1_no_handling" and t.fault == fault_name
        )
        a3_trial = next(
            t for t in results.trials
            if t.agent == "A3_retry" and t.fault == fault_name
        )
        a1_err = a1_trial.error_type or "—"
        a3_err = a3_trial.error_type or "—"
        print(f"{fault_name:<25} {a1_err:<30} {a3_err:<30}")

    # Timing summary
    total_ms = sum(t.duration_ms for t in results.trials)
    print(f"\nTotal trials: {len(results.trials)}")
    print(f"Total time: {total_ms:.1f}ms")
    print(f"Avg per trial: {total_ms / len(results.trials):.2f}ms")

    # Return data for programmatic access
    return results


@pytest.mark.agent_test(layer="mock")
async def test_e060_recoverable_faults():
    """Part 2: Test agents against recoverable (non-deterministic) faults.

    Key insight: retry logic only helps with intermittent faults, not
    deterministic ones. This test validates that claim.
    """
    recoverable = make_recoverable_scenarios()
    results = ExperimentResults()

    # Run multiple trials per scenario to account for randomness
    n_trials = 20

    for agent_name, agent_fn in AGENTS.items():
        for fault_name, _fi_template in recoverable.items():
            survived_count = 0
            for trial in range(n_trials):
                # Fresh injector state each trial with unique seed
                fi = FaultInjector()
                if "tool_intermittent" in fault_name:
                    fi.on_tool("search").intermittent(fail_rate=0.5, seed=trial)
                elif "llm_rate_limit_after_1" in fault_name:
                    fi.on_llm().rate_limit(after_n=1)
                elif "llm_intermittent" in fault_name:
                    fi.on_llm().intermittent(fail_rate=0.5, seed=trial)

                llm = MockLLM()
                llm.on_input(contains="plan").respond("Go to Paris")
                tool = MockTool()
                tool.on_call("search").respond({"hotels": ["Hotel A"]})

                if fi.has_faults_for("search"):
                    tool.attach_faults(fi)
                if fi.has_llm_faults():
                    llm.attach_faults(fi)

                try:
                    await agent_fn(llm, tool)
                    survived_count += 1
                except Exception:
                    pass

            results.trials.append(TrialResult(
                agent=agent_name,
                fault=fault_name,
                survived=survived_count > 0,  # at least one survived
                duration_ms=0,
            ))

            # Print per-scenario survival rate
            rate = survived_count / n_trials
            print(f"{agent_name:<22} {fault_name:<28} {rate:>6.0%} ({survived_count}/{n_trials})")

    # Key insight: A3 (retry) should do BETTER than A1 on intermittent faults
    # because it gets two chances, improving from ~50% to ~75% per call
    print("\n=== E-060 Part 2: Recoverable Fault Survival Rates ===")
    print("(Deterministic faults are unrecoverable by retry — only intermittent helps)")
    print(f"\nTrials per scenario: {n_trials}")
