"""JSON/JSONL file importer for production traces.

Handles two formats:
1. JSON file containing a list of trace objects
2. JSONL file with one trace object per line

Each trace object is normalized to an AgentRun. The importer supports
both CheckAgent-native format (already AgentRun-shaped) and a common
observability format with nested spans/events.

Requirements: F6.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall


class JsonFileImporter:
    """Import traces from JSON or JSONL files."""

    def import_traces(
        self,
        source: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Import traces from a JSON or JSONL file.

        Args:
            source: Path to .json or .jsonl file.
            filters: Optional filters. Supported keys:
                - status: "error" or "success" to filter by outcome
                - tags: list of tags, trace must have at least one
            limit: Max number of traces to return.

        Returns:
            List of AgentRun objects.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {source}")

        raw_traces = self._load_file(path)

        if filters:
            raw_traces = self._apply_filters(raw_traces, filters)

        if limit is not None:
            raw_traces = raw_traces[:limit]

        return [self._normalize(t) for t in raw_traces]

    def _load_file(self, path: Path) -> list[dict[str, Any]]:
        """Load traces from JSON or JSONL file."""
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in text.strip().splitlines()
                if line.strip()
            ]

        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Single trace object
        return [data]

    def _apply_filters(
        self, traces: list[dict[str, Any]], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply key-value filters to raw trace dicts."""
        result = traces

        if "status" in filters:
            target = filters["status"]
            if target == "error":
                result = [t for t in result if t.get("error") is not None]
            elif target == "success":
                result = [t for t in result if t.get("error") is None]

        if "tags" in filters:
            tag_set = set(filters["tags"])
            result = [
                t
                for t in result
                if tag_set & set(t.get("tags", []) or t.get("metadata", {}).get("tags", []))
            ]

        return result

    def _normalize(self, raw: dict[str, Any]) -> AgentRun:
        """Normalize a raw trace dict into an AgentRun.

        Supports two shapes:
        1. CheckAgent-native: has "input", "steps", "final_output" keys
        2. Span-based: has "spans" or "events" with nested tool calls
        """
        # Native format — delegate to Pydantic
        if "input" in raw and "steps" in raw:
            return self._from_native(raw)

        # Span-based format
        if "spans" in raw:
            return self._from_spans(raw)

        # Flat format: minimal trace with input/output
        return self._from_flat(raw)

    def _from_native(self, raw: dict[str, Any]) -> AgentRun:
        """Parse a trace already in AgentRun-compatible format."""
        input_data = raw["input"]
        if isinstance(input_data, str):
            input_data = {"query": input_data}

        steps = []
        for i, s in enumerate(raw.get("steps", [])):
            tool_calls = [
                ToolCall(
                    name=tc.get("name", "unknown"),
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                    error=tc.get("error"),
                    duration_ms=tc.get("duration_ms"),
                )
                for tc in s.get("tool_calls", [])
            ]
            steps.append(
                Step(
                    step_index=s.get("step_index", i),
                    input_text=s.get("input_text"),
                    output_text=s.get("output_text"),
                    tool_calls=tool_calls,
                    model=s.get("model"),
                    prompt_tokens=s.get("prompt_tokens"),
                    completion_tokens=s.get("completion_tokens"),
                    duration_ms=s.get("duration_ms"),
                    metadata=s.get("metadata", {}),
                )
            )

        return AgentRun(
            input=AgentInput(**input_data),
            steps=steps,
            final_output=raw.get("final_output"),
            error=raw.get("error"),
            duration_ms=raw.get("duration_ms"),
            total_prompt_tokens=raw.get("total_prompt_tokens"),
            total_completion_tokens=raw.get("total_completion_tokens"),
            metadata=raw.get("metadata", {}),
        )

    def _from_spans(self, raw: dict[str, Any]) -> AgentRun:
        """Parse a span-based trace (common observability format)."""
        query = raw.get("input", raw.get("query", raw.get("name", "")))
        if isinstance(query, dict):
            query = query.get("query", str(query))

        steps = []
        for i, span in enumerate(raw["spans"]):
            tool_calls = []
            for tc in span.get("tool_calls", span.get("events", [])):
                if isinstance(tc, dict) and ("name" in tc or "tool" in tc):
                    tool_calls.append(
                        ToolCall(
                            name=tc.get("name", tc.get("tool", "unknown")),
                            arguments=tc.get("arguments", tc.get("attributes", {})),
                            result=tc.get("result", tc.get("output")),
                            error=tc.get("error"),
                        )
                    )

            steps.append(
                Step(
                    step_index=i,
                    input_text=span.get("input", span.get("name")),
                    output_text=span.get("output"),
                    tool_calls=tool_calls,
                    model=span.get("model", span.get("attributes", {}).get("model")),
                    prompt_tokens=span.get("prompt_tokens"),
                    completion_tokens=span.get("completion_tokens"),
                    duration_ms=span.get("duration_ms"),
                )
            )

        return AgentRun(
            input=AgentInput(query=query),
            steps=steps,
            final_output=raw.get("output", raw.get("final_output")),
            error=raw.get("error"),
            duration_ms=raw.get("duration_ms"),
            metadata=raw.get("metadata", {}),
        )

    def _from_flat(self, raw: dict[str, Any]) -> AgentRun:
        """Parse a minimal flat trace with just input/output."""
        query = raw.get("input", raw.get("query", raw.get("prompt", "")))
        if isinstance(query, dict):
            query = query.get("query", str(query))

        return AgentRun(
            input=AgentInput(query=query),
            final_output=raw.get("output", raw.get("response", raw.get("final_output"))),
            error=raw.get("error"),
            duration_ms=raw.get("duration_ms"),
            metadata=raw.get("metadata", {}),
        )
