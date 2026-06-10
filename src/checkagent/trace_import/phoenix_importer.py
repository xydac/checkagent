"""Arize Phoenix API trace importer.

Fetches spans from the Arize Phoenix REST API and normalizes them into
AgentRun objects, grouping child spans by their parent trace.

Requirements: F6.2
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall

_DEFAULT_HOST = "http://localhost:6006"
_PAGE_LIMIT = 100


class PhoenixAPIImporter:
    """Import traces from the Arize Phoenix REST API.

    Connects to a running Phoenix instance (local or cloud) and fetches
    spans, grouping them by trace into AgentRun objects.

    Args:
        host: Phoenix base URL (default: http://localhost:6006).
        api_key: Phoenix API key (required for cloud instances, optional locally).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        api_key: str | None = None,
    ) -> None:
        import os

        self._host = host.rstrip("/")
        self._api_key = api_key or os.environ.get("PHOENIX_API_KEY", "")

    def import_traces(
        self,
        source: str = "",
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Fetch traces from the Phoenix API.

        Args:
            source: Ignored — connection info comes from constructor / env vars.
            filters: Optional filters. Supported keys:
                - status: "error" or "success"
            limit: Max number of root traces to return.

        Returns:
            List of AgentRun objects.

        Raises:
            RuntimeError: If the API is unreachable or returns an error.
        """
        spans = self._fetch_spans(limit=limit)
        runs = self._group_into_runs(spans)

        if filters:
            runs = self._apply_filters(runs, filters)

        return runs

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _fetch_spans(self, limit: int | None) -> list[dict[str, Any]]:
        """Paginate through /api/v1/spans."""
        results: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            remaining = None if limit is None else limit - len(results)
            if remaining is not None and remaining <= 0:
                break

            page_size = min(_PAGE_LIMIT, remaining) if remaining is not None else _PAGE_LIMIT
            params: dict[str, Any] = {"limit": page_size}
            if cursor:
                params["cursor"] = cursor
            url = f"{self._host}/api/v1/spans?{urllib.parse.urlencode(params)}"

            data = self._get_json(url)
            page_spans = data.get("data", [])
            if not page_spans:
                break

            results.extend(page_spans)

            cursor = data.get("next")
            if not cursor:
                break

        return results

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Phoenix API error {exc.code}: {body[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Phoenix connection error ({self._host}): {exc.reason}. "
                "Is Phoenix running? Start with: python -m phoenix.server.main serve"
            ) from exc

    def _group_into_runs(self, spans: list[dict[str, Any]]) -> list[AgentRun]:
        """Group spans by trace ID and build one AgentRun per root span."""
        by_trace: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            trace_id = span.get("context", {}).get("trace_id") or span.get("traceId", "unknown")
            by_trace.setdefault(trace_id, []).append(span)

        runs = []
        for trace_id, trace_spans in by_trace.items():
            run = self._spans_to_run(trace_id, trace_spans)
            if run is not None:
                runs.append(run)
        return runs

    def _spans_to_run(
        self, trace_id: str, spans: list[dict[str, Any]]
    ) -> AgentRun | None:
        if not spans:
            return None

        # Root span has no parentId
        root = next(
            (
                s for s in spans
                if not s.get("parentId") and not s.get("context", {}).get("parent_id")
            ),
            spans[0],
        )
        children = [s for s in spans if s is not root]

        # Extract query from root span
        root_input = root.get("input", {})
        if isinstance(root_input, dict):
            val = root_input.get("value", root_input)
            if isinstance(val, dict):
                query = val.get("query") or val.get("input") or str(val)
            else:
                query = str(val)
        else:
            query = str(root_input) if root_input else root.get("name", "")

        # Build steps from child spans
        steps: list[Step] = []
        for i, child in enumerate(children):
            attrs = child.get("attributes", {})
            span_kind = child.get("spanKind", "").upper()
            child_name = child.get("name", "")

            tool_calls: list[ToolCall] = []
            if span_kind == "TOOL" or "tool" in child_name.lower():
                child_input = child.get("input", {})
                if isinstance(child_input, dict):
                    args = child_input.get("value", child_input)
                    if not isinstance(args, dict):
                        args = {}
                else:
                    args = {}
                child_output = child.get("output", {})
                if isinstance(child_output, dict):
                    result = child_output.get("value")
                else:
                    result = str(child_output)
                status_ok = child.get("statusCode", "OK") == "OK"
                tool_calls.append(
                    ToolCall(
                        name=child_name,
                        arguments=args,
                        result=str(result) if result is not None else None,
                        error=child.get("statusMessage") if not status_ok else None,
                        duration_ms=_span_ms(child),
                    )
                )

            child_input_str = _extract_text(child.get("input"))
            child_output_str = _extract_text(child.get("output"))

            steps.append(
                Step(
                    step_index=i,
                    input_text=child_input_str,
                    output_text=child_output_str,
                    tool_calls=tool_calls,
                    model=attrs.get("llm.model_name") or attrs.get("llm.model"),
                    prompt_tokens=_safe_int(attrs.get("llm.token_count.prompt")),
                    completion_tokens=_safe_int(attrs.get("llm.token_count.completion")),
                    duration_ms=_span_ms(child),
                )
            )

        root_output = root.get("output", {})
        final_output = _extract_text(root_output)

        status_code = root.get("statusCode", "OK")
        error = root.get("statusMessage") or "Error" if status_code != "OK" else None

        return AgentRun(
            input=AgentInput(query=query),
            steps=steps,
            final_output=final_output,
            error=error,
            duration_ms=_span_ms(root),
            metadata={
                "trace_id": trace_id,
                "span_name": root.get("name", ""),
                "source": "phoenix",
            },
        )

    def _apply_filters(self, runs: list[AgentRun], filters: dict[str, Any]) -> list[AgentRun]:
        result = runs
        if "status" in filters:
            if filters["status"] == "error":
                result = [r for r in result if r.error is not None]
            elif filters["status"] == "success":
                result = [r for r in result if r.error is None]
        return result


def _extract_text(value: Any) -> str | None:
    """Pull a text string from Phoenix's {value: ...} input/output shape."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        v = value.get("value", value)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("output") or v.get("text") or v.get("content") or str(v)
        return str(v)
    return str(value)


def _span_ms(span: dict[str, Any]) -> float | None:
    """Compute span duration in ms from ISO timestamps or numeric fields."""
    start = span.get("startTime")
    end = span.get("endTime")
    if start and end:
        try:
            from datetime import datetime, timezone

            def _parse(s: str) -> datetime:
                s = s.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(s)
                except ValueError:
                    clean = s.replace("+00:00", "")
                    return datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S.%f").replace(
                        tzinfo=timezone.utc
                    )

            delta = _parse(end) - _parse(start)
            return delta.total_seconds() * 1000
        except Exception:
            pass
    return span.get("duration_ms")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
