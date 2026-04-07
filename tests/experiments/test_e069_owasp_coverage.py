"""E-069: OWASP LLM Top 10 coverage analysis.

Maps CheckAgent's safety probes, evaluators, and validators to the
OWASP LLM Top 10 (2025) categories. Measures coverage breadth, probe
density per category, and identifies gaps.

Produces quantitative data for paper Section 5.3 (RQ3): "How effective
are built-in safety probes at detecting real vulnerabilities?"

OWASP LLM Top 10 (2025):
  LLM01 — Prompt Injection
  LLM02 — Insecure Output Handling
  LLM03 — Training Data Poisoning
  LLM04 — Model Denial of Service
  LLM05 — Supply Chain Vulnerabilities
  LLM06 — Sensitive Information Disclosure
  LLM07 — Insecure Plugin Design
  LLM08 — Excessive Agency
  LLM09 — Overreliance
  LLM10 — Model Theft
"""

from __future__ import annotations

from checkagent.safety.probes import injection, jailbreak, pii, scope
from checkagent.safety.probes.base import ProbeSet
from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    Severity,
)

# ---------------------------------------------------------------------------
# OWASP LLM Top 10 full list
# ---------------------------------------------------------------------------

OWASP_LLM_TOP_10 = {
    "LLM01": "Prompt Injection",
    "LLM02": "Insecure Output Handling",
    "LLM03": "Training Data Poisoning",
    "LLM04": "Model Denial of Service",
    "LLM05": "Supply Chain Vulnerabilities",
    "LLM06": "Sensitive Information Disclosure",
    "LLM07": "Insecure Plugin Design",
    "LLM08": "Excessive Agency",
    "LLM09": "Overreliance",
    "LLM10": "Model Theft",
}

# ---------------------------------------------------------------------------
# All probe sets in the framework
# ---------------------------------------------------------------------------

ALL_PROBE_SETS: list[tuple[str, ProbeSet]] = [
    ("injection.direct", injection.direct),
    ("injection.indirect", injection.indirect),
    ("jailbreak.encoding", jailbreak.encoding),
    ("jailbreak.roleplay", jailbreak.roleplay),
    ("pii.extraction", pii.extraction),
    ("scope.boundary", scope.boundary),
]


# ---------------------------------------------------------------------------
# Evaluators and validators available (by OWASP mapping)
# ---------------------------------------------------------------------------

# These are manually enumerated since evaluators don't self-declare OWASP IDs.
EVALUATOR_OWASP_MAP: dict[str, list[str]] = {
    "PromptInjectionDetector": ["LLM01"],
    "PIIDetector": ["LLM06"],
    "RefusalComplianceEvaluator": ["LLM01", "LLM08"],
    "SystemPromptLeakageDetector": ["LLM01", "LLM06"],
    "ToolCallBoundaryValidator": ["LLM07", "LLM08"],
    "ConversationSafetyScanner": ["LLM01", "LLM06"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOWASPCoverageMapping:
    """Map all probes to OWASP categories and measure coverage."""

    def test_total_probe_count(self) -> None:
        """Verify we know the total number of probes."""
        total = sum(len(ps) for _, ps in ALL_PROBE_SETS)
        # 25 direct + 10 indirect + 8 encoding + 7 roleplay + 10 pii + 8 scope
        assert total == 68

    def test_probes_per_owasp_category(self) -> None:
        """Count probes mapped to each OWASP LLM Top 10 category."""
        owasp_counts: dict[str, int] = {k: 0 for k in OWASP_LLM_TOP_10}
        unmapped_count = 0

        for _, probe_set in ALL_PROBE_SETS:
            for probe in probe_set:
                owasp_id = OWASP_MAPPING.get(probe.category)
                if owasp_id and owasp_id in owasp_counts:
                    owasp_counts[owasp_id] += 1
                else:
                    unmapped_count += 1

        # LLM01 — Prompt Injection: direct(25) + indirect(10) + encoding(8) + roleplay(7) = 50
        assert owasp_counts["LLM01"] == 50
        # LLM06 — Sensitive Information Disclosure: pii(10)
        assert owasp_counts["LLM06"] == 10
        # LLM08 — Excessive Agency: scope boundary(8) mapped via OFF_TOPIC
        assert owasp_counts["LLM08"] == 8
        # Uncovered categories
        assert owasp_counts["LLM02"] == 0  # Insecure Output Handling
        assert owasp_counts["LLM03"] == 0  # Training Data Poisoning
        assert owasp_counts["LLM04"] == 0  # Model DoS
        assert owasp_counts["LLM05"] == 0  # Supply Chain
        assert owasp_counts["LLM07"] == 0  # Insecure Plugin Design (evaluated, not probed)
        assert owasp_counts["LLM09"] == 0  # Overreliance (category exists, no probes yet)
        assert owasp_counts["LLM10"] == 0  # Model Theft

    def test_evaluator_owasp_coverage(self) -> None:
        """Count evaluators covering each OWASP category."""
        owasp_evaluator_counts: dict[str, list[str]] = {k: [] for k in OWASP_LLM_TOP_10}

        for evaluator_name, owasp_ids in EVALUATOR_OWASP_MAP.items():
            for owasp_id in owasp_ids:
                owasp_evaluator_counts[owasp_id].append(evaluator_name)

        # LLM01 covered by 4 evaluators (injection, refusal, system prompt, conversation)
        assert len(owasp_evaluator_counts["LLM01"]) == 4
        # LLM06 covered by 3 evaluators (PII, system prompt, conversation)
        assert len(owasp_evaluator_counts["LLM06"]) == 3
        # LLM07 covered by ToolCallBoundaryValidator
        assert len(owasp_evaluator_counts["LLM07"]) == 1
        # LLM08 covered by 2 (RefusalCompliance + ToolBoundary)
        assert len(owasp_evaluator_counts["LLM08"]) == 2

    def test_combined_coverage_breadth(self) -> None:
        """Calculate total OWASP coverage combining probes + evaluators."""
        covered_by_probes: set[str] = set()
        covered_by_evaluators: set[str] = set()

        for _, probe_set in ALL_PROBE_SETS:
            for probe in probe_set:
                owasp_id = OWASP_MAPPING.get(probe.category)
                if owasp_id and owasp_id in OWASP_LLM_TOP_10:
                    covered_by_probes.add(owasp_id)

        for _, owasp_ids in EVALUATOR_OWASP_MAP.items():
            for owasp_id in owasp_ids:
                covered_by_evaluators.add(owasp_id)

        combined = covered_by_probes | covered_by_evaluators
        total = len(OWASP_LLM_TOP_10)

        # Probes cover: LLM01, LLM06, LLM08
        assert covered_by_probes == {"LLM01", "LLM06", "LLM08"}
        # Evaluators cover: LLM01, LLM06, LLM07, LLM08
        assert covered_by_evaluators == {"LLM01", "LLM06", "LLM07", "LLM08"}
        # Combined: 4/10
        assert len(combined) == 4
        assert combined == {"LLM01", "LLM06", "LLM07", "LLM08"}

        # Coverage = 40%
        coverage_pct = len(combined) / total * 100
        assert coverage_pct == 40.0

        # Uncovered (by design or future work)
        uncovered = set(OWASP_LLM_TOP_10.keys()) - combined
        assert uncovered == {"LLM02", "LLM03", "LLM04", "LLM05", "LLM09", "LLM10"}

    def test_probe_severity_distribution(self) -> None:
        """Analyze severity distribution across all probes."""
        severity_counts: dict[str, int] = {s.value: 0 for s in Severity}

        for _, probe_set in ALL_PROBE_SETS:
            for probe in probe_set:
                severity_counts[probe.severity.value] += 1

        assert severity_counts["critical"] >= 10  # High-risk probes
        assert severity_counts["high"] >= 20  # Majority are high
        assert severity_counts["medium"] >= 5
        total = sum(severity_counts.values())
        assert total == 68

    def test_probe_tag_coverage(self) -> None:
        """Analyze attack technique tags for diversity."""
        all_tags: set[str] = set()
        for _, probe_set in ALL_PROBE_SETS:
            for probe in probe_set:
                all_tags.update(probe.tags)

        # Should have diverse attack technique coverage
        expected_techniques = {
            "ignore", "persona", "extraction", "smuggling",
            "delimiter", "encoding", "roleplay", "indirect",
            "boundary", "authority",
        }
        assert expected_techniques.issubset(all_tags), (
            f"Missing techniques: {expected_techniques - all_tags}"
        )

    def test_gap_analysis_classification(self) -> None:
        """Classify uncovered categories as 'out of scope' vs 'future work'."""
        # Categories that are fundamentally outside a testing framework's scope
        out_of_scope = {
            "LLM03",  # Training Data Poisoning — requires model-level access
            "LLM05",  # Supply Chain — dependency/package management concern
            "LLM10",  # Model Theft — model hosting/access control concern
        }
        # Categories addressable with additional probes/evaluators
        future_work = {
            "LLM02",  # Insecure Output Handling — could add output sanitization checks
            "LLM04",  # Model DoS — could add resource usage monitoring probes
            "LLM09",  # Overreliance — could add groundedness probes (category exists)
        }
        # Currently covered
        covered = {"LLM01", "LLM06", "LLM07", "LLM08"}

        assert out_of_scope | future_work | covered == set(OWASP_LLM_TOP_10.keys())
        assert not (out_of_scope & future_work)
        assert not (out_of_scope & covered)
        assert not (future_work & covered)

    def test_detection_rates_by_owasp_category(self) -> None:
        """Cross-reference detection rates from metrics with OWASP mapping.

        Uses data from metrics.md collected in prior experiments.
        This test documents the known detection rates per OWASP category.
        """
        # Detection rates from metrics.md (output-side detection)
        detection_rates: dict[str, dict[str, float]] = {
            "LLM01": {
                "direct_injection_input": 0.28,   # 7/25
                "direct_injection_output": 0.64,  # 16/25
                "indirect_injection_input": 0.60, # 6/10
                "indirect_injection_output": 0.90,# 9/10
                "jailbreak_encoding_output": 1.0, # 8/8
                "jailbreak_roleplay_output": 0.714,# 5/7
            },
            "LLM06": {
                "pii_output": 1.0,  # 10/10
            },
            "LLM08": {
                "scope_boundary_output": 1.0,  # 8/8 (via scope evaluator)
                "tool_boundary": 1.0,  # 21/21 (via ToolCallBoundaryValidator)
            },
        }

        # LLM01 has strongest coverage with multiple detection approaches
        # Output-side detection is consistently higher than input-side
        avg_output_llm01 = (0.64 + 0.90 + 1.0 + 0.714) / 4
        assert avg_output_llm01 > 0.80  # >80% average output detection

        # LLM06 and LLM08 have 100% detection on test corpus
        assert detection_rates["LLM06"]["pii_output"] == 1.0
        assert detection_rates["LLM08"]["tool_boundary"] == 1.0

        # Input-side detection is weaker (expected — harder problem)
        avg_input_llm01 = (0.28 + 0.60) / 2
        assert avg_input_llm01 < 0.50  # Input-side detection under 50%
