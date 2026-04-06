"""OpenTelemetry OTLP JSON importer for production traces.

Parses OTLP JSON export format and normalizes spans into AgentRun objects.
Groups spans by trace ID, identifies root spans as agent runs, and extracts
tool calls from child spans with "tool" or "function" in their name.

Requirements: F6.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall


class OtelJsonImporter:
    """Import traces from OpenTelemetry OTLP JSON export files."""

    def import_traces(
        self,
        source: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Import traces from an OTLP JSON file.

        Args:
            source: Path to OTLP JSON file.
            filters: Optional filters. Supported keys:
                - status: "error" to filter for errored traces only
            limit: Max number of traces to return.

        Returns:
            List of AgentRun objects.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"OTel trace file not found: {source}")

        data = json.loads(path.read_text(encoding="utf-8"))
        spans = self._extract_spans(data)
        grouped = self._group_by_trace(spans)

        runs = []
        for trace_id, trace_spans in grouped.items():
            run = self._trace_to_agent_run(trace_id, trace_spans)
            if run is not None:
                runs.append(run)

        if filters:
            runs = self._apply_filters(runs, filters)

        if limit is not None:
            runs = runs[:limit]

        return runs

    def _extract_spans(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract flat list of spans from OTLP JSON structure."""
        spans = []

        # OTLP JSON has resourceSpans -> scopeSpans -> spans
        for rs in data.get("resourceSpans", data.get("resource_spans", [])):
            for ss in rs.get("scopeSpans", rs.get("scope_spans", [])):
                for span in ss.get("spans", []):
                    spans.append(span)

        # Also handle flat list of spans
        if not spans and isinstance(data.get("spans"), list):
            spans = data["spans"]

        return spans

    def _group_by_trace(
        self, spans: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group spans by traceId."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            trace_id = span.get("traceId", span.get("trace_id", "unknown"))
            groups.setdefault(trace_id, []).append(span)
        return groups

    def _trace_to_agent_run(
        self, trace_id: str, spans: list[dict[str, Any]]
    ) -> AgentRun | None:
        """Convert a group of spans from one trace into an AgentRun."""
        if not spans:
            return None

        # Find root span (no parentSpanId or empty)
        root = None
        children = []
        for span in spans:
            parent = span.get("parentSpanId", span.get("parent_span_id", ""))
            if not parent:
                root = span
            else:
                children.append(span)

        if root is None:
            root = spans[0]
            children = spans[1:]

        # Extract input from root span attributes
        attrs = self._flatten_attributes(root.get("attributes", []))
        query = (
            attrs.get("input")
            or attrs.get("llm.input")
            or attrs.get("agent.input")
            or root.get("name", "")
        )

        # Build steps from child spans
        steps = []
        for i, child in enumerate(children):
            child_attrs = self._flatten_attributes(child.get("attributes", []))
            child_name = child.get("name", "")

            tool_calls = []
            is_tool = any(
                kw in child_name.lower()
                for kw in ("tool", "function", "action")
            )
            if is_tool:
                tool_calls.append(
                    ToolCall(
                        name=child_name,
                        arguments=_safe_json(child_attrs.get("tool.arguments", "{}")),
                        result=child_attrs.get("tool.result", child_attrs.get("output")),
                        error=child_attrs.get("error"),
                        duration_ms=self._span_duration_ms(child),
                    )
                )

            steps.append(
                Step(
                    step_index=i,
                    input_text=child_attrs.get("input", child_name),
                    output_text=child_attrs.get("output"),
                    tool_calls=tool_calls,
                    model=child_attrs.get("llm.model", child_attrs.get("model")),
                    prompt_tokens=_safe_int(child_attrs.get("llm.prompt_tokens")),
                    completion_tokens=_safe_int(child_attrs.get("llm.completion_tokens")),
                    duration_ms=self._span_duration_ms(child),
                )
            )

        # Check for error status
        status = root.get("status", {})
        error = None
        if status.get("code") == 2 or status.get("statusCode") == "ERROR":
            error = status.get("message", "Unknown error")

        output = attrs.get("output") or attrs.get("llm.output") or attrs.get("agent.output")

        return AgentRun(
            input=AgentInput(query=query),
            steps=steps,
            final_output=output,
            error=error,
            duration_ms=self._span_duration_ms(root),
            metadata={"trace_id": trace_id, "source": "otel"},
        )

    def _flatten_attributes(
        self, attributes: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, Any]:
        """Flatten OTLP attribute arrays into a simple dict.

        OTLP attributes come as [{key: "k", value: {stringValue: "v"}}].
        Also handles already-flattened dicts.
        """
        if isinstance(attributes, dict):
            return attributes

        result = {}
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value", {})
            if isinstance(value, dict):
                # OTLP value types: stringValue, intValue, boolValue, etc.
                for vtype in ("stringValue", "intValue", "boolValue", "doubleValue"):
                    if vtype in value:
                        result[key] = value[vtype]
                        break
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _span_duration_ms(self, span: dict[str, Any]) -> float | None:
        """Calculate span duration in milliseconds from start/end times."""
        start = span.get("startTimeUnixNano", span.get("start_time_unix_nano"))
        end = span.get("endTimeUnixNano", span.get("end_time_unix_nano"))
        if start is not None and end is not None:
            return (int(end) - int(start)) / 1_000_000
        return span.get("duration_ms")

    def _apply_filters(
        self, runs: list[AgentRun], filters: dict[str, Any]
    ) -> list[AgentRun]:
        """Apply filters to imported runs."""
        result = runs
        if "status" in filters:
            if filters["status"] == "error":
                result = [r for r in result if r.error is not None]
            elif filters["status"] == "success":
                result = [r for r in result if r.error is None]
        return result


def _safe_json(value: Any) -> dict[str, Any]:
    """Try to parse a JSON string into a dict, return empty dict on failure."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _safe_int(value: Any) -> int | None:
    """Try to convert a value to int, return None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
