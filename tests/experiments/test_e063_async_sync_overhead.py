"""E-063: Async vs sync adapter performance overhead.

Planned experiment E-002: Compare throughput of async adapter (native) vs
sync adapter (thread executor) for the same agent logic. Quantify the cost
of sync wrapping via run_in_executor.

RQ4 (cost-performance): What overhead does the adapter layer add?
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from checkagent import GenericAdapter

# --- Agent implementations (identical logic, different signatures) ---

def sync_simple_agent(query: str) -> str:
    """Sync agent: trivial string transform."""
    return query.upper()


async def async_simple_agent(query: str) -> str:
    """Async agent: trivial string transform (no real I/O)."""
    return query.upper()


def sync_complex_agent(query: str) -> str:
    """Sync agent: simulate multi-step processing."""
    tokens = query.split()
    # Simulate tool calls / processing steps
    results = []
    for token in tokens:
        results.append(token.upper())
        results.append(token[::-1])
    return " ".join(results)


async def async_complex_agent(query: str) -> str:
    """Async agent: same processing, async signature."""
    tokens = query.split()
    results = []
    for token in tokens:
        results.append(token.upper())
        results.append(token[::-1])
    return " ".join(results)


WARMUP = 10
ITERATIONS = 200


async def _measure_adapter(adapter: GenericAdapter, query: str) -> list[float]:
    """Run adapter ITERATIONS times, return list of durations in ms."""
    # Warmup
    for _ in range(WARMUP):
        await adapter.run(query)

    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = await adapter.run(query)
        elapsed = (time.perf_counter() - start) * 1000
        assert result.final_output is not None
        times.append(elapsed)
    return times


async def _measure_raw_async(fn, query: str) -> list[float]:
    """Measure raw async function calls without adapter wrapping."""
    for _ in range(WARMUP):
        await fn(query)

    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = await fn(query)
        elapsed = (time.perf_counter() - start) * 1000
        assert result is not None
        times.append(elapsed)
    return times


async def _measure_raw_sync(fn, query: str) -> list[float]:
    """Measure raw sync function calls (run_in_executor for fairness)."""
    loop = asyncio.get_running_loop()
    for _ in range(WARMUP):
        await loop.run_in_executor(None, fn, query)

    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = await loop.run_in_executor(None, fn, query)
        elapsed = (time.perf_counter() - start) * 1000
        assert result is not None
        times.append(elapsed)
    return times


def _stats(times: list[float]) -> dict[str, float]:
    """Compute summary statistics."""
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(len(times) * 0.95)],
        "p99": sorted(times)[int(len(times) * 0.99)],
        "stdev": statistics.stdev(times),
    }


@pytest.mark.agent_test(layer="mock")
async def test_e063_async_adapter_overhead():
    """Measure async adapter overhead vs raw async call."""
    query = "hello world this is a test query"
    adapter = GenericAdapter(async_simple_agent)

    raw_times = await _measure_raw_async(async_simple_agent, query)
    adapter_times = await _measure_adapter(adapter, query)

    raw = _stats(raw_times)
    wrapped = _stats(adapter_times)

    overhead_pct = ((wrapped["median"] - raw["median"]) / raw["median"]) * 100

    # Record for experiment log
    print("\n--- E-063: Async Simple Agent ---")
    print(f"Raw async:    median={raw['median']:.3f}ms  p95={raw['p95']:.3f}ms")
    print(f"Adapter:      median={wrapped['median']:.3f}ms  p95={wrapped['p95']:.3f}ms")
    print(f"Overhead:     {overhead_pct:.1f}%")

    # Assertion: adapter overhead should be reasonable
    # The key assertion is absolute time — must stay under 100ms per test
    assert wrapped["p99"] < 100, f"p99 exceeds 100ms mock-layer target: {wrapped['p99']:.3f}ms"


@pytest.mark.agent_test(layer="mock")
async def test_e063_sync_adapter_overhead():
    """Measure sync adapter (thread executor) overhead vs raw executor call."""
    query = "hello world this is a test query"
    adapter = GenericAdapter(sync_simple_agent)

    raw_times = await _measure_raw_sync(sync_simple_agent, query)
    adapter_times = await _measure_adapter(adapter, query)

    raw = _stats(raw_times)
    wrapped = _stats(adapter_times)

    overhead_pct = ((wrapped["median"] - raw["median"]) / raw["median"]) * 100

    print("\n--- E-063: Sync Simple Agent ---")
    print(f"Raw executor: median={raw['median']:.3f}ms  p95={raw['p95']:.3f}ms")
    print(f"Adapter:      median={wrapped['median']:.3f}ms  p95={wrapped['p95']:.3f}ms")
    print(f"Overhead:     {overhead_pct:.1f}%")

    assert wrapped["p99"] < 100, f"p99 exceeds 100ms mock-layer target: {wrapped['p99']:.3f}ms"


@pytest.mark.agent_test(layer="mock")
async def test_e063_async_vs_sync_comparison():
    """Compare async-native vs sync-wrapped adapter performance."""
    query = "hello world this is a test query with more tokens"
    async_adapter = GenericAdapter(async_simple_agent)
    sync_adapter = GenericAdapter(sync_simple_agent)

    async_times = await _measure_adapter(async_adapter, query)
    sync_times = await _measure_adapter(sync_adapter, query)

    a = _stats(async_times)
    s = _stats(sync_times)

    sync_penalty_pct = ((s["median"] - a["median"]) / a["median"]) * 100

    print("\n--- E-063: Async vs Sync Adapter ---")
    print(f"Async adapter: median={a['median']:.3f}ms  p95={a['p95']:.3f}ms  p99={a['p99']:.3f}ms")
    print(f"Sync adapter:  median={s['median']:.3f}ms  p95={s['p95']:.3f}ms  p99={s['p99']:.3f}ms")
    print(f"Sync penalty:  {sync_penalty_pct:.1f}%")

    # Both must stay under 100ms mock-layer target
    assert a["p99"] < 100, f"Async p99 exceeds 100ms: {a['p99']:.3f}ms"
    assert s["p99"] < 100, f"Sync p99 exceeds 100ms: {s['p99']:.3f}ms"


@pytest.mark.agent_test(layer="mock")
async def test_e063_complex_agent_overhead():
    """Measure overhead on a more complex agent with multi-step processing."""
    query = "the quick brown fox jumps over the lazy dog"
    async_adapter = GenericAdapter(async_complex_agent)
    sync_adapter = GenericAdapter(sync_complex_agent)

    async_times = await _measure_adapter(async_adapter, query)
    sync_times = await _measure_adapter(sync_adapter, query)

    a = _stats(async_times)
    s = _stats(sync_times)

    sync_penalty_pct = ((s["median"] - a["median"]) / a["median"]) * 100

    print("\n--- E-063: Complex Agent Async vs Sync ---")
    print(f"Async adapter: median={a['median']:.3f}ms  p95={a['p95']:.3f}ms  p99={a['p99']:.3f}ms")
    print(f"Sync adapter:  median={s['median']:.3f}ms  p95={s['p95']:.3f}ms  p99={s['p99']:.3f}ms")
    print(f"Sync penalty:  {sync_penalty_pct:.1f}%")

    assert a["p99"] < 100
    assert s["p99"] < 100


@pytest.mark.agent_test(layer="mock")
async def test_e063_adapter_wrapping_overhead_isolated():
    """Isolate pure adapter overhead: type construction + timing."""
    query = "test"

    # Compare raw call vs Adapter.run() (includes AgentInput + Step + AgentRun)
    adapter = GenericAdapter(async_simple_agent)

    raw_times = await _measure_raw_async(async_simple_agent, query)
    adapter_times = await _measure_adapter(adapter, query)

    raw = _stats(raw_times)
    wrapped = _stats(adapter_times)

    # The difference is the "adapter tax": type construction + timing
    adapter_tax_ms = wrapped["median"] - raw["median"]
    overhead_pct = (adapter_tax_ms / raw["median"]) * 100 if raw["median"] > 0 else 0

    print("\n--- E-063: Adapter Tax (Isolated) ---")
    print(f"Raw call:     median={raw['median']:.4f}ms")
    print(f"Adapter call: median={wrapped['median']:.4f}ms")
    print(f"Adapter tax:  {adapter_tax_ms:.4f}ms ({overhead_pct:.1f}%)")
    print("Components:   AgentInput coercion + perf_counter + Step + AgentRun")

    # Adapter tax should be < 1ms for trivial agents
    assert adapter_tax_ms < 1.0, f"Adapter tax {adapter_tax_ms:.4f}ms exceeds 1ms budget"


@pytest.mark.agent_test(layer="mock")
async def test_e063_throughput_at_scale():
    """Measure sustained throughput: how many agent runs per second?"""
    query = "benchmark query"
    async_adapter = GenericAdapter(async_simple_agent)
    sync_adapter = GenericAdapter(sync_simple_agent)

    count = 1000

    # Warmup
    for _ in range(WARMUP):
        await async_adapter.run(query)
        await sync_adapter.run(query)

    # Async throughput
    start = time.perf_counter()
    for _ in range(count):
        await async_adapter.run(query)
    async_elapsed = time.perf_counter() - start
    async_rps = count / async_elapsed

    # Sync throughput
    start = time.perf_counter()
    for _ in range(count):
        await sync_adapter.run(query)
    sync_elapsed = time.perf_counter() - start
    sync_rps = count / sync_elapsed

    print(f"\n--- E-063: Throughput ({count} runs) ---")
    print(f"Async: {async_rps:.0f} runs/sec ({async_elapsed:.3f}s total)")
    print(f"Sync:  {sync_rps:.0f} runs/sec ({sync_elapsed:.3f}s total)")
    print(f"Ratio: async is {async_rps/sync_rps:.1f}x faster")

    # Both should sustain > 100 runs/sec for trivial agents
    assert async_rps > 100, f"Async throughput too low: {async_rps:.0f} rps"
    assert sync_rps > 100, f"Sync throughput too low: {sync_rps:.0f} rps"
