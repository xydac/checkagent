"""Test case generator — convert AgentRun traces into golden dataset test cases.

Takes imported production traces and generates EvalCase entries suitable
for inclusion in a golden dataset for regression testing.

Requirements: F6.2
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from checkagent.core.types import AgentRun
from checkagent.datasets.schema import EvalCase, GoldenDataset
from checkagent.trace_import.pii import PiiScrubber


def generate_test_cases(
    runs: list[AgentRun],
    *,
    scrub_pii: bool = True,
    pii_scrubber: PiiScrubber | None = None,
    dataset_name: str = "imported",
    tags: list[str] | None = None,
) -> GoldenDataset:
    """Generate a golden dataset from imported AgentRun traces.

    Args:
        runs: List of AgentRun objects to convert.
        scrub_pii: Whether to scrub PII from inputs/outputs.
        pii_scrubber: Custom PiiScrubber instance. Uses default if None.
        dataset_name: Name for the generated dataset.
        tags: Additional tags to add to all generated test cases.

    Returns:
        A GoldenDataset containing one EvalCase per run.
    """
    scrubber = pii_scrubber or PiiScrubber() if scrub_pii else None
    extra_tags = tags or []

    cases = []
    for run in runs:
        if scrubber:
            scrubber.reset()

        case = _run_to_test_case(run, scrubber=scrubber, extra_tags=extra_tags)
        cases.append(case)

    return GoldenDataset(
        name=dataset_name,
        version="1",
        description=f"Auto-generated from {len(runs)} imported production traces",
        cases=cases,
    )


def _run_to_test_case(
    run: AgentRun,
    *,
    scrubber: PiiScrubber | None,
    extra_tags: list[str],
) -> EvalCase:
    """Convert a single AgentRun into a EvalCase."""
    query = run.input.query
    if scrubber:
        query = scrubber.scrub_text(query)

    # Generate a deterministic ID from the input
    case_id = _generate_id(query)

    # Extract expected tool sequence
    expected_tools = [tc.name for tc in run.tool_calls]

    # Extract output patterns for matching
    expected_output_contains: list[str] = []
    if run.final_output and isinstance(run.final_output, str):
        output = run.final_output
        if scrubber:
            output = scrubber.scrub_text(output)
        # Take first meaningful sentence as expected output pattern
        sentences = [s.strip() for s in output.split(".") if len(s.strip()) > 10]
        if sentences:
            expected_output_contains = [sentences[0]]

    # Compute tags
    tags = list(extra_tags)
    tags.append("imported")
    if run.error:
        tags.append("error")
    if run.tool_calls:
        tags.append("has-tools")

    # Build context from run metadata
    context: dict[str, Any] = {}
    if run.input.context:
        context.update(
            scrubber.scrub_value(run.input.context) if scrubber else run.input.context
        )

    # Build metadata
    metadata: dict[str, Any] = {}
    if run.duration_ms is not None:
        metadata["original_duration_ms"] = run.duration_ms
    if run.total_tokens is not None:
        metadata["original_total_tokens"] = run.total_tokens
    if run.error:
        metadata["original_error"] = (
            scrubber.scrub_text(run.error) if scrubber else run.error
        )

    return EvalCase(
        id=case_id,
        input=query,
        expected_tools=expected_tools,
        expected_output_contains=expected_output_contains,
        max_steps=max(len(run.steps), 1) * 2 if run.steps else None,
        tags=tags,
        context=context,
        metadata=metadata,
    )


def _generate_id(query: str) -> str:
    """Generate a short deterministic ID from the query text."""
    h = hashlib.sha256(query.encode()).hexdigest()[:8]
    # Create a readable prefix from the query
    words = query.split()[:3]
    prefix = "-".join(w.lower()[:8] for w in words if w.isalnum() or w.replace("-", "").isalnum())
    prefix = prefix[:20] if prefix else "trace"
    return f"{prefix}-{h}"


def export_dataset_yaml(dataset: GoldenDataset, path: str) -> None:
    """Export a golden dataset to a YAML file.

    Uses JSON-compatible output since checkagent prefers JSON,
    but writes in a YAML-friendly format for human readability.
    """
    import yaml  # type: ignore[import-untyped]  # noqa: F811

    data = json.loads(dataset.model_dump_json())
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def export_dataset_json(dataset: GoldenDataset, path: str) -> None:
    """Export a golden dataset to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))
