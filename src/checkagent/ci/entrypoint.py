"""CI entrypoint for quality gate evaluation and report generation.

Called by the GitHub Action after pytest completes. Parses JUnit XML,
evaluates quality gates from checkagent.yml, and writes a Markdown summary.

Usage:
    python -m checkagent.ci.entrypoint --junit results.xml --output-dir report/
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from checkagent.ci.quality_gate import evaluate_gates
from checkagent.ci.reporter import RunSummary, generate_pr_comment
from checkagent.core.config import load_config


def parse_junit_xml(path: Path) -> RunSummary:
    """Parse a JUnit XML file into a RunSummary."""
    tree = ET.parse(path)  # noqa: S314
    root = tree.getroot()

    # Handle both <testsuites> wrapper and bare <testsuite>
    suites = list(root) if root.tag == "testsuites" else [root]

    total = 0
    failures = 0
    errors = 0
    skipped = 0
    duration = 0.0

    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        duration += float(suite.get("time", 0))

    passed = total - failures - errors - skipped

    return RunSummary(
        total=total,
        passed=passed,
        failed=failures,
        errors=errors,
        skipped=skipped,
        duration_s=duration,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for CI quality gate evaluation."""
    parser = argparse.ArgumentParser(description="CheckAgent CI entrypoint")
    parser.add_argument("--junit", type=Path, help="Path to JUnit XML results")
    parser.add_argument("--output-dir", type=Path, default=Path("checkagent-report"))
    parser.add_argument("--config", type=Path, default=None, help="Path to checkagent.yml")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(argv)

    # Load config
    config = load_config(args.config)

    # Parse test results
    test_summary = None
    if args.junit and args.junit.exists():
        test_summary = parse_junit_xml(args.junit)

    # Evaluate quality gates
    gate_report = None
    if config.quality_gates and test_summary is not None:
        scores = {"pass_rate": test_summary.pass_rate}
        gate_report = evaluate_gates(scores, config.quality_gates)

    # Generate report
    comment = generate_pr_comment(
        test_summary=test_summary,
        gate_report=gate_report,
    )

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(comment, encoding="utf-8")

    print(f"Report written to {summary_path}")

    # Exit with failure if gates blocked
    if gate_report is not None and not gate_report.passed:
        print("Quality gates BLOCKED — failing CI")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
