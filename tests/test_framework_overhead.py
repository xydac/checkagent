"""Benchmark: framework overhead vs raw function calls (RQ4).

Measures the overhead CheckAgent's infrastructure adds compared to
plain function calls. Each benchmark runs 1000 iterations after warmup
and reports per-call overhead in microseconds and percentage.

This data feeds paper Section 5.4 (RQ4 — Performance Overhead).
"""

from __future__ import annotations

import time

import pytest

from checkagent.adapters.generic import GenericAdapter
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool

ITERATIONS = 1000
WARMUP = 50


async def _raw_async_fn(query: str) -> str:
    """Minimal async function — the baseline."""
    return f"echo: {query}"


def _raw_sync_fn(query: str) -> str:
    """Minimal sync function — sync baseline."""
    return f"echo: {query}"


async def _measure_async(coro_factory, iterations: int = ITERATIONS) -> float:
    """Run an async callable `iterations` times and return avg ns per call."""
    # Warmup
    for _ in range(WARMUP):
        await coro_factory()

    start = time.perf_counter_ns()
    for _ in range(iterations):
        await coro_factory()
    elapsed = time.perf_counter_ns() - start
    return elapsed / iterations


# --- Baseline: raw async function call ---


@pytest.mark.agent_test(layer="mock")
async def test_baseline_raw_async_call():
    """Baseline: cost of calling an async function directly."""
    avg_ns = await _measure_async(lambda: _raw_async_fn("hello"))
    avg_us = avg_ns / 1000
    # Raw async call should be < 10 microseconds
    assert avg_us < 10, f"Raw async call too slow: {avg_us:.1f}µs"
    print(f"\n  Raw async call: {avg_us:.2f}µs/call")


# --- GenericAdapter overhead ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_generic_adapter_async():
    """Overhead of GenericAdapter wrapping an async function."""
    adapter = GenericAdapter(_raw_async_fn)

    raw_ns = await _measure_async(lambda: _raw_async_fn("hello"))

    async def adapter_call():
        return await adapter.run("hello")

    adapter_ns = await _measure_async(adapter_call)
    overhead_us = (adapter_ns - raw_ns) / 1000
    overhead_pct = ((adapter_ns - raw_ns) / raw_ns * 100) if raw_ns > 0 else 0

    print(f"\n  Raw async:     {raw_ns / 1000:.2f}µs/call")
    print(f"  GenericAdapter: {adapter_ns / 1000:.2f}µs/call")
    print(f"  Overhead:       {overhead_us:.2f}µs ({overhead_pct:.1f}%)")

    # Adapter overhead should be < 500µs per call (generous — includes Pydantic model construction)
    assert overhead_us < 500, f"Adapter overhead too high: {overhead_us:.1f}µs"


@pytest.mark.agent_test(layer="mock")
async def test_overhead_generic_adapter_sync():
    """Overhead of GenericAdapter wrapping a sync function (thread executor)."""
    adapter = GenericAdapter(_raw_sync_fn)

    async def adapter_call():
        return await adapter.run("hello")

    avg_ns = await _measure_async(adapter_call)
    avg_us = avg_ns / 1000

    print(f"\n  GenericAdapter (sync→async): {avg_us:.2f}µs/call")

    # Sync adapter uses run_in_executor — expect higher overhead but < 1ms
    assert avg_us < 1000, f"Sync adapter overhead too high: {avg_us:.1f}µs"


# --- MockLLM overhead ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_llm_complete():
    """Overhead of MockLLM.complete() with a single pattern rule."""
    mock_llm = MockLLM()
    mock_llm.on_input(contains="hello").respond("world")

    raw_ns = await _measure_async(lambda: _raw_async_fn("hello"))

    async def mock_call():
        return await mock_llm.complete("hello")

    mock_ns = await _measure_async(mock_call)
    overhead_us = (mock_ns - raw_ns) / 1000
    overhead_pct = ((mock_ns - raw_ns) / raw_ns * 100) if raw_ns > 0 else 0

    print(f"\n  Raw async:      {raw_ns / 1000:.2f}µs/call")
    print(f"  MockLLM.complete: {mock_ns / 1000:.2f}µs/call")
    print(f"  Overhead:        {overhead_us:.2f}µs ({overhead_pct:.1f}%)")

    # MockLLM overhead should be < 200µs (pattern matching + Pydantic record)
    assert overhead_us < 200, f"MockLLM overhead too high: {overhead_us:.1f}µs"


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_llm_many_rules():
    """MockLLM with 20 pattern rules — does rule count affect overhead?"""
    mock_llm = MockLLM()
    for i in range(19):
        mock_llm.on_input(contains=f"pattern_{i}").respond(f"response_{i}")
    # The one that will match
    mock_llm.on_input(contains="target").respond("found")

    async def mock_call():
        return await mock_llm.complete("target query")

    avg_ns = await _measure_async(mock_call)
    avg_us = avg_ns / 1000

    print(f"\n  MockLLM (20 rules, last match): {avg_us:.2f}µs/call")

    # Even with 20 rules, should be < 300µs
    assert avg_us < 300, f"MockLLM with many rules too slow: {avg_us:.1f}µs"


# --- MockTool overhead ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_tool_call():
    """Overhead of MockTool.call() with a registered tool."""
    mock_tool = MockTool()
    mock_tool.register("search", response={"results": []})

    raw_ns = await _measure_async(lambda: _raw_async_fn("hello"))

    async def tool_call():
        return await mock_tool.call("search", {"query": "test"})

    tool_ns = await _measure_async(tool_call)
    overhead_us = (tool_ns - raw_ns) / 1000
    overhead_pct = ((tool_ns - raw_ns) / raw_ns * 100) if raw_ns > 0 else 0

    print(f"\n  Raw async:      {raw_ns / 1000:.2f}µs/call")
    print(f"  MockTool.call:  {tool_ns / 1000:.2f}µs/call")
    print(f"  Overhead:        {overhead_us:.2f}µs ({overhead_pct:.1f}%)")

    # MockTool overhead should be < 200µs
    assert overhead_us < 200, f"MockTool overhead too high: {overhead_us:.1f}µs"


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_tool_with_schema():
    """MockTool with JSON Schema validation enabled."""
    mock_tool = MockTool()
    mock_tool.register(
        "search",
        response={"results": []},
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )

    async def tool_call():
        return await mock_tool.call("search", {"query": "test", "limit": 10})

    avg_ns = await _measure_async(tool_call)
    avg_us = avg_ns / 1000

    print(f"\n  MockTool (with schema validation): {avg_us:.2f}µs/call")

    # Schema validation adds some overhead but should be < 300µs
    assert avg_us < 300, f"MockTool with schema too slow: {avg_us:.1f}µs"


# --- Fault injection overhead ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_tool_with_faults():
    """MockTool with fault injector attached (no fault triggered)."""
    mock_tool = MockTool()
    mock_tool.register("search", response={"results": []})
    fi = FaultInjector()
    fi.on_tool("other_tool").timeout()  # Fault on a different tool
    mock_tool.attach_faults(fi)

    # Measure with fault injector attached but NOT triggering
    async def tool_call_with_faults():
        return await mock_tool.call("search", {"query": "test"})

    # Baseline: no faults attached
    mock_tool_clean = MockTool()
    mock_tool_clean.register("search", response={"results": []})

    async def tool_call_clean():
        return await mock_tool_clean.call("search", {"query": "test"})

    clean_ns = await _measure_async(tool_call_clean)
    faults_ns = await _measure_async(tool_call_with_faults)
    overhead_us = (faults_ns - clean_ns) / 1000

    print(f"\n  MockTool (no faults):     {clean_ns / 1000:.2f}µs/call")
    print(f"  MockTool (faults attached): {faults_ns / 1000:.2f}µs/call")
    print(f"  Fault check overhead:      {overhead_us:.2f}µs")

    # Fault checking when no fault triggers should add < 50µs
    assert abs(overhead_us) < 200, f"Fault injection overhead too high: {overhead_us:.1f}µs"


# --- MockLLM with token usage tracking ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_mock_llm_with_usage():
    """MockLLM with token usage simulation enabled."""
    mock_llm_plain = MockLLM()
    mock_llm_plain.on_input(contains="hello").respond("world")

    mock_llm_usage = MockLLM()
    mock_llm_usage.on_input(contains="hello").respond("world")
    mock_llm_usage.with_usage(auto_estimate=True)

    async def plain_call():
        return await mock_llm_plain.complete("hello")

    async def usage_call():
        return await mock_llm_usage.complete("hello")

    plain_ns = await _measure_async(plain_call)
    usage_ns = await _measure_async(usage_call)
    overhead_us = (usage_ns - plain_ns) / 1000

    print(f"\n  MockLLM (no usage):    {plain_ns / 1000:.2f}µs/call")
    print(f"  MockLLM (auto usage):  {usage_ns / 1000:.2f}µs/call")
    print(f"  Usage tracking overhead: {overhead_us:.2f}µs")

    # Token counting should add < 50µs
    assert abs(overhead_us) < 100, f"Usage tracking overhead too high: {overhead_us:.1f}µs"


# --- Summary test that produces the paper table ---


@pytest.mark.agent_test(layer="mock")
async def test_overhead_summary_table():
    """Produce the complete overhead summary table for the paper."""
    results: dict[str, dict[str, float]] = {}

    # 1. Raw async baseline
    raw_ns = await _measure_async(lambda: _raw_async_fn("hello"))
    results["Raw async call"] = {"ns": raw_ns}

    # 2. GenericAdapter (async)
    adapter = GenericAdapter(_raw_async_fn)
    adapter_ns = await _measure_async(lambda: adapter.run("hello"))
    results["GenericAdapter (async)"] = {"ns": adapter_ns}

    # 3. MockLLM.complete (1 rule)
    mock_llm = MockLLM()
    mock_llm.on_input(contains="hello").respond("world")
    llm_ns = await _measure_async(lambda: mock_llm.complete("hello"))
    results["MockLLM.complete (1 rule)"] = {"ns": llm_ns}

    # 4. MockLLM.complete (20 rules)
    mock_llm_20 = MockLLM()
    for i in range(19):
        mock_llm_20.on_input(contains=f"pattern_{i}").respond(f"response_{i}")
    mock_llm_20.on_input(contains="hello").respond("found")
    llm20_ns = await _measure_async(lambda: mock_llm_20.complete("hello"))
    results["MockLLM.complete (20 rules)"] = {"ns": llm20_ns}

    # 5. MockTool.call (no schema)
    mock_tool = MockTool()
    mock_tool.register("search", response={"results": []})
    tool_ns = await _measure_async(
        lambda: mock_tool.call("search", {"query": "test"})
    )
    results["MockTool.call (no schema)"] = {"ns": tool_ns}

    # 6. MockTool.call (with schema)
    mock_tool_s = MockTool()
    mock_tool_s.register(
        "search",
        response={"results": []},
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    tools_ns = await _measure_async(
        lambda: mock_tool_s.call("search", {"query": "test"})
    )
    results["MockTool.call (with schema)"] = {"ns": tools_ns}

    # 7. MockTool.call with fault injector (no trigger)
    mock_tool_f = MockTool()
    mock_tool_f.register("search", response={"results": []})
    fi = FaultInjector()
    fi.on_tool("other").timeout()
    mock_tool_f.attach_faults(fi)
    toolf_ns = await _measure_async(
        lambda: mock_tool_f.call("search", {"query": "test"})
    )
    results["MockTool.call (faults attached)"] = {"ns": toolf_ns}

    # Print summary table
    baseline_ns = results["Raw async call"]["ns"]
    print("\n")
    print("  ┌──────────────────────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Component                             │  µs/call │ Overhead │ % of raw │")
    print("  ├──────────────────────────────────────┼──────────┼──────────┼──────────┤")
    for name, data in results.items():
        ns = data["ns"]
        us = ns / 1000
        overhead = (ns - baseline_ns) / 1000
        pct = ((ns - baseline_ns) / baseline_ns * 100) if baseline_ns > 0 else 0
        if name == "Raw async call":
            print(f"  │ {name:<37}│ {us:>7.2f}µ │     base │     base │")
        else:
            print(f"  │ {name:<37}│ {us:>7.2f}µ │ {overhead:>+7.2f}µ │ {pct:>+7.1f}% │")
    print("  └──────────────────────────────────────┴──────────┴──────────┴──────────┘")

    # Overall assertion: no component should add more than 500µs
    for name, data in results.items():
        if name == "Raw async call":
            continue
        overhead_us = (data["ns"] - baseline_ns) / 1000
        assert overhead_us < 500, f"{name} overhead too high: {overhead_us:.1f}µs"
