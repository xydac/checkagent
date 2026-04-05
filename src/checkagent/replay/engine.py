"""Replay engine — matches outbound calls against recorded cassette data.

The engine loads a cassette and serves recorded responses when an outbound
call matches. Supports three matching strategies: exact, subset, and
sequence-based.

Implements F2.2 (Replay) from the PRD.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from checkagent.replay.cassette import Cassette, Interaction, RecordedRequest


class MatchStrategy(str, Enum):
    """How to match outbound calls against recorded interactions."""

    EXACT = "exact"  # Full request body must match exactly
    SUBSET = "subset"  # Recorded body keys must be present in request
    SEQUENCE = "sequence"  # Match by position in call sequence


class CassetteMismatchError(Exception):
    """Raised when no recorded interaction matches an outbound call."""

    def __init__(self, request: RecordedRequest, attempted: str = "") -> None:
        self.request = request
        details = f" (strategy: {attempted})" if attempted else ""
        super().__init__(
            f"No cassette match for {request.kind} call "
            f"'{request.method}'{details}"
        )


class ReplayEngine:
    """Serves recorded responses from a cassette.

    Usage::

        engine = ReplayEngine(cassette)
        response = engine.match(RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={"messages": [...]},
        ))
        # response is the RecordedResponse from the cassette
    """

    def __init__(
        self,
        cassette: Cassette,
        strategy: MatchStrategy = MatchStrategy.EXACT,
        *,
        block_unmatched: bool = True,
    ) -> None:
        self._cassette = cassette
        self._strategy = strategy
        self._block_unmatched = block_unmatched
        self._sequence_index = 0
        self._used: set[int] = set()

    @property
    def cassette(self) -> Cassette:
        """The loaded cassette."""
        return self._cassette

    @property
    def remaining(self) -> int:
        """Number of unused interactions."""
        return len(self._cassette.interactions) - len(self._used)

    @property
    def all_used(self) -> bool:
        """Whether all recorded interactions have been consumed."""
        return len(self._used) == len(self._cassette.interactions)

    def match(self, request: RecordedRequest) -> Interaction:
        """Find a matching recorded interaction for the given request.

        Raises CassetteMismatchError if no match is found and
        block_unmatched is True. Returns None-safe (always raises or
        returns a valid Interaction).
        """
        if self._strategy == MatchStrategy.SEQUENCE:
            return self._match_sequence(request)
        elif self._strategy == MatchStrategy.EXACT:
            return self._match_exact(request)
        elif self._strategy == MatchStrategy.SUBSET:
            return self._match_subset(request)
        raise ValueError(f"Unknown strategy: {self._strategy}")

    def _match_sequence(self, request: RecordedRequest) -> Interaction:
        """Match by position in the interaction sequence."""
        idx = self._sequence_index
        if idx >= len(self._cassette.interactions):
            if self._block_unmatched:
                raise CassetteMismatchError(request, "sequence")
            raise CassetteMismatchError(request, "sequence")
        interaction = self._cassette.interactions[idx]
        self._sequence_index += 1
        self._used.add(idx)
        return interaction

    def _match_exact(self, request: RecordedRequest) -> Interaction:
        """Match by exact method + body equality."""
        for idx, interaction in enumerate(self._cassette.interactions):
            if idx in self._used:
                continue
            rec = interaction.request
            if (
                rec.kind == request.kind
                and rec.method == request.method
                and self._bodies_equal(rec.body, request.body)
            ):
                self._used.add(idx)
                return interaction

        if self._block_unmatched:
            raise CassetteMismatchError(request, "exact")
        raise CassetteMismatchError(request, "exact")

    def _match_subset(self, request: RecordedRequest) -> Interaction:
        """Match by kind + method, and recorded body keys are a subset."""
        for idx, interaction in enumerate(self._cassette.interactions):
            if idx in self._used:
                continue
            rec = interaction.request
            if rec.kind != request.kind or rec.method != request.method:
                continue
            if self._is_subset(rec.body, request.body):
                self._used.add(idx)
                return interaction

        if self._block_unmatched:
            raise CassetteMismatchError(request, "subset")
        raise CassetteMismatchError(request, "subset")

    @staticmethod
    def _bodies_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """Compare two bodies by their canonical JSON."""
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )

    @staticmethod
    def _is_subset(
        recorded: dict[str, Any], actual: dict[str, Any]
    ) -> bool:
        """Check if all keys in recorded exist in actual with same values."""
        for key, value in recorded.items():
            if key not in actual:
                return False
            if isinstance(value, dict) and isinstance(actual[key], dict):
                if not ReplayEngine._is_subset(value, actual[key]):
                    return False
            elif value != actual[key]:
                return False
        return True

    def reset(self) -> None:
        """Reset the engine to replay the cassette from the beginning."""
        self._sequence_index = 0
        self._used.clear()
