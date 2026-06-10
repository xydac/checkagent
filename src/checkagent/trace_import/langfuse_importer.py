"""Langfuse API trace importer.

Fetches traces from the Langfuse REST API and normalizes them into
AgentRun objects for use with CheckAgent's test generation pipeline.

Requirements: F6.2
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall

_DEFAULT_HOST = "https://cloud.langfuse.com"
_PAGE_LIMIT = 50


class LangfuseAPIImporter:
    """Import traces from the Langfuse REST API.

    Credentials are read from the constructor or from environment variables
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.

    Args:
        host: Langfuse base URL (default: https://cloud.langfuse.com).
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        import os

        self._host = host.rstrip("/")
        self._public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")

    def import_traces(
        self,
        source: str = "",
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Fetch traces from the Langfuse API.

        Args:
            source: Ignored — connection info comes from constructor / env vars.
            filters: Optional filters. Supported keys:
                - status: "error" or "success"
            limit: Max number of traces to return (fetched page by page).

        Returns:
            List of AgentRun objects.

        Raises:
            RuntimeError: If credentials are missing or the API returns an error.
        """
        if not self._public_key or not self._secret_key:
            raise RuntimeError(
                "Langfuse credentials required. Set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY environment variables, or pass public_key "
                "and secret_key to LangfuseAPIImporter."
            )

        raw_traces = self._fetch_all(limit=limit)

        if filters:
            raw_traces = self._apply_filters(raw_traces, filters)

        return [self._normalize(t) for t in raw_traces]

    def _auth_header(self) -> str:
        creds = f"{self._public_key}:{self._secret_key}"
        encoded = base64.b64encode(creds.encode()).decode()
        return f"Basic {encoded}"

    def _fetch_all(self, limit: int | None) -> list[dict[str, Any]]:
        """Paginate through /api/public/traces until limit or end of data."""
        results: list[dict[str, Any]] = []
        page = 1

        while True:
            remaining = None if limit is None else limit - len(results)
            if remaining is not None and remaining <= 0:
                break

            page_size = min(_PAGE_LIMIT, remaining) if remaining is not None else _PAGE_LIMIT
            params = urllib.parse.urlencode({"page": page, "limit": page_size})
            url = f"{self._host}/api/public/traces?{params}"

            data = self._get_json(url)
            page_data = data.get("data", [])
            if not page_data:
                break

            results.extend(page_data)

            meta = data.get("meta", {})
            total_pages = meta.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1

        if limit is not None:
            results = results[:limit]

        return results

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={"Authorization": self._auth_header(), "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Langfuse API error {exc.code}: {body[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Langfuse connection error: {exc.reason}") from exc

    def _apply_filters(
        self, traces: list[dict[str, Any]], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        result = traces
        if "status" in filters:
            if filters["status"] == "error":
                result = [
                    t for t in result
                    if t.get("metadata", {}).get("error") or not t.get("output")
                ]
            elif filters["status"] == "success":
                result = [t for t in result if t.get("output") is not None]
        return result

    def _normalize(self, raw: dict[str, Any]) -> AgentRun:
        """Convert a Langfuse trace object into an AgentRun."""
        inp = raw.get("input", "")
        if isinstance(inp, dict):
            if inp.get("messages"):
                first_msg = inp.get("messages", [{}])[0]
                query = first_msg.get("content", str(inp))
            else:
                query = inp.get("query") or str(inp)
        else:
            query = str(inp) if inp else ""

        observations = raw.get("observations", [])
        steps: list[Step] = []
        for i, obs in enumerate(observations):
            obs_type = obs.get("type", "").upper()
            obs_input = obs.get("input", {})
            obs_output = obs.get("output")

            tool_calls: list[ToolCall] = []
            if obs_type == "SPAN" and obs.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=obs.get("name", "unknown"),
                        arguments=obs_input if isinstance(obs_input, dict) else {},
                        result=str(obs_output) if obs_output is not None else None,
                        duration_ms=_ms(obs.get("latency")),
                    )
                )

            usage = obs.get("usage") or obs.get("usageDetails") or {}
            steps.append(
                Step(
                    step_index=i,
                    input_text=str(obs_input) if obs_input else None,
                    output_text=str(obs_output) if obs_output is not None else None,
                    tool_calls=tool_calls,
                    model=obs.get("model"),
                    prompt_tokens=usage.get("input") or usage.get("promptTokens"),
                    completion_tokens=usage.get("output") or usage.get("completionTokens"),
                    duration_ms=_ms(obs.get("latency")),
                )
            )

        output = raw.get("output")
        if isinstance(output, dict):
            output = output.get("text") or output.get("content") or str(output)

        return AgentRun(
            input=AgentInput(query=query),
            steps=steps,
            final_output=str(output) if output is not None else None,
            duration_ms=_ms(raw.get("latency")),
            metadata={
                "trace_id": raw.get("id", ""),
                "trace_name": raw.get("name", ""),
                "source": "langfuse",
            },
        )


def _ms(latency: Any) -> float | None:
    """Convert Langfuse latency (seconds as float) to milliseconds."""
    if latency is None:
        return None
    try:
        return float(latency) * 1000
    except (ValueError, TypeError):
        return None
