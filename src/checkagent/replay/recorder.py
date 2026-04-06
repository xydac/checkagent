"""Cassette recorder — captures LLM and tool interactions during live runs.

The recorder accumulates interactions as they happen, then finalizes
and saves the cassette. Designed to wrap around an agent adapter's run
method to transparently capture all calls.

Implements F2.1 (Recording) from the PRD.
"""

from __future__ import annotations

import time
from typing import Any

from checkagent.replay.cassette import (
    Cassette,
    CassetteMeta,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    redact_dict,
)


class CassetteRecorder:
    """Records LLM and tool interactions into a cassette.

    Usage::

        recorder = CassetteRecorder(test_id="tests/test_agent::test_hello")
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [...]},
            response_body={"choices": [...]},
            prompt_tokens=10,
            completion_tokens=20,
        )
        cassette = recorder.finalize()
        cassette.save(Path("cassettes/test_hello/abc123.json"))
    """

    def __init__(
        self,
        test_id: str = "",
        redact_keys: frozenset[str] | None = None,
    ) -> None:
        self._test_id = test_id
        self._redact_keys = redact_keys
        self._interactions: list[Interaction] = []
        self._finalized = False

    @property
    def interaction_count(self) -> int:
        """Number of recorded interactions."""
        return len(self._interactions)

    def record_llm_call(
        self,
        method: str,
        request_body: dict[str, Any],
        response_body: Any,
        *,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: float | None = None,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> Interaction:
        """Record an LLM API call (request + response)."""
        if self._finalized:
            raise RuntimeError("Cannot record after finalize()")

        body = self._redact(request_body)
        interaction = Interaction(
            request=RecordedRequest(
                kind="llm",
                method=method,
                model=model,
                body=body,
            ),
            response=RecordedResponse(
                status=status,
                body=response_body,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
            ),
            metadata=metadata or {},
        )
        self._interactions.append(interaction)
        return interaction

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        duration_ms: float | None = None,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> Interaction:
        """Record a tool invocation (call + result)."""
        if self._finalized:
            raise RuntimeError("Cannot record after finalize()")

        body = self._redact(arguments)
        interaction = Interaction(
            request=RecordedRequest(
                kind="tool",
                method=tool_name,
                body=body,
            ),
            response=RecordedResponse(
                status=status,
                body=result,
                duration_ms=duration_ms,
            ),
            metadata=metadata or {},
        )
        self._interactions.append(interaction)
        return interaction

    def finalize(self) -> Cassette:
        """Build and finalize the cassette with all recorded interactions."""
        cassette = Cassette(
            meta=CassetteMeta(test_id=self._test_id),
            interactions=list(self._interactions),
        )
        cassette.finalize()
        self._finalized = True
        return cassette

    def _redact(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply redaction to request body if redact keys are configured."""
        if self._redact_keys is not None:
            return redact_dict(body, self._redact_keys)
        return redact_dict(body)


class TimedCall:
    """Context manager for timing an API call.

    Usage::

        with TimedCall() as tc:
            response = await client.chat(...)
        recorder.record_llm_call(..., duration_ms=tc.duration_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> TimedCall:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.duration_ms = (time.perf_counter() - self._start) * 1000
