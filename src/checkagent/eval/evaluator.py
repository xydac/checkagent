"""Custom evaluator plugin interface.

Provides the base class for user-defined evaluation metrics.
Evaluators can be registered via entry points or in conftest.py.

Requirements: F3.5
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from checkagent.core.types import AgentRun, Score
from checkagent.datasets.schema import TestCase


class Evaluator(ABC):
    """Base class for custom evaluators.

    Subclass this to define team-specific metrics. Evaluators are
    discovered via the ``checkagent.evaluators`` entry point group
    or by registering in conftest.py.

    Example::

        class ResponseToneEvaluator(Evaluator):
            name = "response_tone"

            def score(self, run: AgentRun, expected: TestCase) -> Score:
                output = str(run.final_output or "")
                is_polite = any(w in output.lower() for w in ["please", "thank"])
                return Score(
                    name=self.name,
                    value=1.0 if is_polite else 0.0,
                    threshold=0.5,
                )
    """

    name: str = "unnamed_evaluator"

    @abstractmethod
    def score(self, run: AgentRun, expected: TestCase) -> Score:
        """Evaluate an agent run against a test case.

        Args:
            run: The completed agent run trace.
            expected: The golden test case with expected outcomes.

        Returns:
            A Score with value between 0.0 and 1.0.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class EvaluatorRegistry:
    """Registry for custom evaluators.

    Collects evaluators from entry points and manual registration,
    then runs them all against agent runs.
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        """Register a custom evaluator."""
        self._evaluators[evaluator.name] = evaluator

    def unregister(self, name: str) -> None:
        """Remove an evaluator by name."""
        self._evaluators.pop(name, None)

    @property
    def evaluators(self) -> dict[str, Evaluator]:
        """All registered evaluators."""
        return dict(self._evaluators)

    def discover_entry_points(self, group: str = "checkagent.evaluators") -> int:
        """Load evaluators from installed entry points.

        Args:
            group: Entry point group to scan.

        Returns:
            Number of evaluators discovered.
        """
        from importlib.metadata import entry_points

        eps = entry_points(group=group)
        count = 0
        for ep in eps:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Evaluator):
                instance = cls()
                self.register(instance)
                count += 1
        return count

    def score_all(
        self, run: AgentRun, expected: TestCase
    ) -> dict[str, Score]:
        """Run all registered evaluators against an agent run.

        Args:
            run: The completed agent run trace.
            expected: The golden test case.

        Returns:
            Dict mapping evaluator name to Score.
        """
        results: dict[str, Score] = {}
        for name, evaluator in self._evaluators.items():
            results[name] = evaluator.score(run, expected)
        return results

    def __len__(self) -> int:
        return len(self._evaluators)

    def __contains__(self, name: str) -> bool:
        return name in self._evaluators
