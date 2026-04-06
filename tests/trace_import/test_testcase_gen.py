"""Tests for test case generator."""

from __future__ import annotations

import json

import pytest

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.trace_import.testcase_gen import (
    export_dataset_json,
    generate_test_cases,
)


def _make_run(
    query="Test query",
    steps=None,
    final_output="Test output",
    error=None,
    duration_ms=None,
    total_prompt_tokens=None,
    total_completion_tokens=None,
):
    return AgentRun(
        input=AgentInput(query=query),
        steps=steps or [],
        final_output=final_output,
        error=error,
        duration_ms=duration_ms,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
    )


class TestGenerateTestCases:
    def test_basic_generation(self):
        runs = [_make_run(query="Hello agent")]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert len(dataset.cases) == 1
        assert dataset.cases[0].input == "Hello agent"
        assert "imported" in dataset.cases[0].tags

    def test_expected_tools_extracted(self):
        runs = [
            _make_run(
                steps=[
                    Step(
                        step_index=0,
                        tool_calls=[
                            ToolCall(name="search", arguments={"q": "test"}),
                            ToolCall(name="summarize"),
                        ],
                    )
                ]
            )
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert dataset.cases[0].expected_tools == ["search", "summarize"]

    def test_error_tag_added(self):
        runs = [_make_run(error="Something failed")]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert "error" in dataset.cases[0].tags

    def test_has_tools_tag(self):
        runs = [
            _make_run(
                steps=[
                    Step(
                        tool_calls=[ToolCall(name="search")],
                    )
                ]
            )
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert "has-tools" in dataset.cases[0].tags

    def test_custom_tags_included(self):
        runs = [_make_run()]
        dataset = generate_test_cases(
            runs, scrub_pii=False, tags=["regression", "prod"]
        )
        tags = dataset.cases[0].tags
        assert "regression" in tags
        assert "prod" in tags
        assert "imported" in tags

    def test_dataset_name(self):
        runs = [_make_run()]
        dataset = generate_test_cases(
            runs, scrub_pii=False, dataset_name="my-traces"
        )
        assert dataset.name == "my-traces"

    def test_pii_scrubbed_in_query(self):
        runs = [_make_run(query="Contact john@example.com")]
        dataset = generate_test_cases(runs, scrub_pii=True)
        assert "john@example.com" not in dataset.cases[0].input
        assert "<EMAIL_1>" in dataset.cases[0].input

    def test_pii_scrubbed_in_error(self):
        runs = [_make_run(error="Failed for user john@example.com")]
        dataset = generate_test_cases(runs, scrub_pii=True)
        meta = dataset.cases[0].metadata
        assert "john@example.com" not in meta.get("original_error", "")

    def test_max_steps_computed(self):
        runs = [
            _make_run(
                steps=[Step(step_index=0), Step(step_index=1), Step(step_index=2)]
            )
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        # max_steps = len(steps) * 2 = 6
        assert dataset.cases[0].max_steps == 6

    def test_duration_in_metadata(self):
        runs = [_make_run(duration_ms=1500.0)]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert dataset.cases[0].metadata["original_duration_ms"] == 1500.0

    def test_token_count_in_metadata(self):
        runs = [
            _make_run(total_prompt_tokens=100, total_completion_tokens=50)
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert dataset.cases[0].metadata["original_total_tokens"] == 150

    def test_duplicate_queries_raise(self):
        with pytest.raises(Exception, match="Duplicate"):
            generate_test_cases(
                [_make_run(query="Same"), _make_run(query="Same")],
                scrub_pii=False,
            )

    def test_unique_queries_unique_ids(self):
        runs = [
            _make_run(query="Query one"),
            _make_run(query="Query two"),
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        ids = [c.id for c in dataset.cases]
        assert len(set(ids)) == 2

    def test_expected_output_contains(self):
        runs = [
            _make_run(
                final_output="The refund has been initiated. "
                "Please allow 3-5 business days for processing."
            )
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        # Should extract first meaningful sentence
        assert len(dataset.cases[0].expected_output_contains) > 0

    def test_export_json(self, tmp_path):
        runs = [_make_run(query="Export test")]
        dataset = generate_test_cases(runs, scrub_pii=False)
        out = tmp_path / "output.json"
        export_dataset_json(dataset, str(out))

        loaded = json.loads(out.read_text())
        assert loaded["name"] == "imported"
        assert len(loaded["cases"]) == 1
        assert loaded["cases"][0]["input"] == "Export test"

    def test_multiple_runs(self):
        runs = [
            _make_run(query=f"Query {i}", final_output=f"Output {i}")
            for i in range(5)
        ]
        dataset = generate_test_cases(runs, scrub_pii=False)
        assert len(dataset.cases) == 5
        assert dataset.description == "Auto-generated from 5 imported production traces"
