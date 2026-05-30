"""Scan performance benchmarks — locks in overhead contracts for CI and the paper.

These tests capture the quantitative data from E-168 (Cycle 168):
  - 101-probe full scan: 830ms avg wall time (Linux)
  - Per-probe average: 8.2ms/probe
  - checkagent demo command: 2.17s wall time
  - Semaphore concurrency bound: 10 concurrent probes

All assertions include 7–12x headroom over measured baselines to tolerate
slow CI runners while still catching 5x+ regressions.

RQ4 (framework overhead) reference: see checkagent-research/experiments.md E-168.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest

from checkagent.cli.scan import _PROBE_SETS, _run_all_probes

# ---------------------------------------------------------------------------
# Shared fixture: fast in-memory agent (no LLM calls, deterministic)
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"


def _build_agent_fn() -> object:
    """Return the demo agent callable (assets/demo_agent.py:my_agent)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_agent", ASSETS_DIR / "demo_agent.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.my_agent


def _all_probes() -> list:
    """Return the full built-in probe list (all categories)."""
    probes = []
    for cat_probes in _PROBE_SETS.values():
        probes.extend(cat_probes)
    return probes


# ---------------------------------------------------------------------------
# Test 1 — full scan wall time  (E-168: 830ms avg on Linux)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_scan_wall_time() -> None:
    """101-probe scan must complete in under 10 seconds on any CI runner.

    # E-168: measured 830ms avg on Linux; 10s gives 12x headroom for slow CI runners.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_agent", ASSETS_DIR / "demo_agent.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    agent_fn = mod.my_agent

    probes = _all_probes()
    assert len(probes) >= 100, f"Expected at least 100 probes, got {len(probes)}"

    t0 = time.monotonic()
    asyncio.run(_run_all_probes(agent_fn, probes, timeout=5.0))
    elapsed = time.monotonic() - t0

    # E-168: measured 830ms avg on Linux; 10s gives 12x headroom for slow CI runners.
    assert elapsed < 10.0, (
        f"Full scan took {elapsed:.2f}s — expected < 10.0s "
        f"(E-168 baseline: 830ms, 12x headroom)"
    )


# ---------------------------------------------------------------------------
# Test 2 — per-probe average  (E-168: 8.2ms/probe)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_per_probe_average_ms() -> None:
    """Per-probe average must stay under 100ms.

    # E-168: measured 8.2ms/probe; 100ms gives 12x headroom.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_agent", ASSETS_DIR / "demo_agent.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    agent_fn = mod.my_agent

    probes = _all_probes()
    n = len(probes)

    t0 = time.monotonic()
    asyncio.run(_run_all_probes(agent_fn, probes, timeout=5.0))
    elapsed = time.monotonic() - t0

    per_probe_ms = (elapsed / n) * 1000

    # E-168: measured 8.2ms/probe; 100ms gives 12x headroom.
    assert per_probe_ms < 100.0, (
        f"Per-probe average: {per_probe_ms:.1f}ms — expected < 100ms "
        f"(E-168 baseline: 8.2ms, 12x headroom)"
    )


# ---------------------------------------------------------------------------
# Test 3 — demo command wall time  (E-168: 2.17s)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_demo_command_wall_time() -> None:
    """checkagent demo must complete in under 15 seconds.

    # E-168: measured 2.17s; 15s gives 7x headroom for CI.
    """
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "checkagent", "demo"],
        capture_output=True,
        timeout=30,
    )
    elapsed = time.monotonic() - t0

    # The demo command may exit non-zero if it finds issues in the demo agent —
    # that is acceptable for a performance test; we only care about wall time.
    # E-168: measured 2.17s; 15s gives 7x headroom for CI.
    assert elapsed < 15.0, (
        f"checkagent demo took {elapsed:.2f}s — expected < 15.0s "
        f"(E-168 baseline: 2.17s, 7x headroom). "
        f"stdout={result.stdout[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — semaphore concurrency bound  (E-168: 10 concurrent probes)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_semaphore_concurrency_bound() -> None:
    """_run_all_probes must use a Semaphore(10) concurrency limit.

    Inspect the source of _run_all_probes to confirm the bound equals 10.
    This ensures the documented E-168 concurrency=10 contract is not silently
    changed, which would affect both throughput and resource usage profiles.
    """
    import inspect

    source = inspect.getsource(_run_all_probes)

    # Confirm Semaphore is used with value 10
    assert "asyncio.Semaphore(10)" in source, (
        "Expected _run_all_probes to use asyncio.Semaphore(10) for concurrency control. "
        "E-168 documents concurrency=10 as the design choice. "
        "If you changed this, update E-168 in checkagent-research/experiments.md."
    )
