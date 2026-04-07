"""Conversation-level safety scanner.

Scans multi-turn conversations for safety issues that span multiple turns.
Per-turn scanning misses split-credential attacks, PII accumulation, and
escalation patterns (see E-064). This scanner runs evaluators on both
individual turns and the aggregate conversation to close those gaps.

Implements the conversation-aware safety evaluation recommended by E-064.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from checkagent.safety.evaluator import SafetyEvaluator, SafetyFinding

if TYPE_CHECKING:
    from checkagent.conversation.session import Conversation


@dataclass
class ConversationSafetyResult:
    """Result of scanning a full conversation for safety issues.

    Attributes
    ----------
    passed : bool
        True only if no findings in any mode (per-turn or aggregate).
    per_turn_findings : dict[int, list[SafetyFinding]]
        Findings keyed by turn index (0-based).
    aggregate_findings : list[SafetyFinding]
        Findings from scanning the concatenated conversation output.
    aggregate_only_findings : list[SafetyFinding]
        Findings that appeared ONLY in aggregate mode — these are the
        split/accumulation attacks that per-turn scanning missed.
    """

    passed: bool
    per_turn_findings: dict[int, list[SafetyFinding]] = field(default_factory=dict)
    aggregate_findings: list[SafetyFinding] = field(default_factory=list)
    aggregate_only_findings: list[SafetyFinding] = field(default_factory=list)
    evaluator: str = "conversation_scanner"
    details: dict[str, object] = field(default_factory=dict)

    @property
    def total_per_turn_findings(self) -> int:
        return sum(len(f) for f in self.per_turn_findings.values())

    @property
    def total_findings(self) -> int:
        return self.total_per_turn_findings + len(self.aggregate_only_findings)

    @property
    def turns_with_findings(self) -> list[int]:
        return sorted(k for k, v in self.per_turn_findings.items() if v)


class ConversationSafetyScanner:
    """Scans multi-turn conversations with multiple safety evaluators.

    Runs each evaluator in two modes:

    1. **Per-turn** — evaluates each turn's output individually.
    2. **Aggregate** — evaluates the concatenated output of all turns.

    Findings that appear only in aggregate mode are flagged as
    ``aggregate_only_findings`` — these represent split/accumulation
    attacks invisible to per-turn scanning.

    Usage::

        scanner = ConversationSafetyScanner([
            PromptInjectionDetector(),
            PIILeakageScanner(),
        ])
        result = scanner.scan(conversation)
        assert result.passed
        # Or check specifically for split attacks:
        assert len(result.aggregate_only_findings) == 0
    """

    def __init__(self, evaluators: list[SafetyEvaluator]) -> None:
        self._evaluators = list(evaluators)

    def scan(self, conversation: Conversation) -> ConversationSafetyResult:
        """Scan the conversation with all registered evaluators."""
        per_turn_findings: dict[int, list[SafetyFinding]] = {}
        all_per_turn_descriptions: set[str] = set()

        # Phase 1: per-turn scanning
        for turn in conversation.turns:
            turn_findings: list[SafetyFinding] = []
            output = turn.output_text
            if output is None:
                continue
            for evaluator in self._evaluators:
                result = evaluator.evaluate(output)
                for finding in result.findings:
                    turn_findings.append(finding)
                    all_per_turn_descriptions.add(finding.description)
            if turn_findings:
                per_turn_findings[turn.index] = turn_findings

        # Phase 2: aggregate scanning
        all_outputs = []
        for turn in conversation.turns:
            output = turn.output_text
            if output is not None:
                all_outputs.append(output)
        combined_text = " ".join(all_outputs)

        aggregate_findings: list[SafetyFinding] = []
        for evaluator in self._evaluators:
            result = evaluator.evaluate(combined_text)
            aggregate_findings.extend(result.findings)

        # Phase 3: identify aggregate-only findings (the split attacks)
        aggregate_only: list[SafetyFinding] = [
            f for f in aggregate_findings
            if f.description not in all_per_turn_descriptions
        ]

        passed = not per_turn_findings and not aggregate_findings

        return ConversationSafetyResult(
            passed=passed,
            per_turn_findings=per_turn_findings,
            aggregate_findings=aggregate_findings,
            aggregate_only_findings=aggregate_only,
            details={
                "total_turns": conversation.total_turns,
                "evaluators": [e.name for e in self._evaluators],
                "per_turn_count": sum(len(f) for f in per_turn_findings.values()),
                "aggregate_count": len(aggregate_findings),
                "aggregate_only_count": len(aggregate_only),
            },
        )
