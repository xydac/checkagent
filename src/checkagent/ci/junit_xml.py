"""JUnit XML output for CI/CD dashboard integration.

Generates standard JUnit XML format compatible with Jenkins, CircleCI,
Azure DevOps, GitHub Actions, and other CI systems.

Implements PRD requirement F5.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.sax.saxutils import escape


@dataclass
class JUnitProperty:
    """A key-value property attached to a test suite or test case."""

    name: str
    value: str


@dataclass
class JUnitTestCase:
    """A single test case result."""

    name: str
    classname: str
    time_s: float = 0.0
    failure_message: str | None = None
    failure_text: str | None = None
    error_message: str | None = None
    error_text: str | None = None
    skipped_message: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    properties: list[JUnitProperty] = field(default_factory=list)

    @property
    def is_failure(self) -> bool:
        return self.failure_message is not None

    @property
    def is_error(self) -> bool:
        return self.error_message is not None

    @property
    def is_skipped(self) -> bool:
        return self.skipped_message is not None

    @property
    def is_passed(self) -> bool:
        return not self.is_failure and not self.is_error and not self.is_skipped


@dataclass
class JUnitTestSuite:
    """A collection of test cases (maps to a pytest test file or class)."""

    name: str
    test_cases: list[JUnitTestCase] = field(default_factory=list)
    timestamp: str | None = None
    hostname: str | None = None
    properties: list[JUnitProperty] = field(default_factory=list)

    @property
    def tests(self) -> int:
        return len(self.test_cases)

    @property
    def failures(self) -> int:
        return sum(1 for tc in self.test_cases if tc.is_failure)

    @property
    def errors(self) -> int:
        return sum(1 for tc in self.test_cases if tc.is_error)

    @property
    def skipped(self) -> int:
        return sum(1 for tc in self.test_cases if tc.is_skipped)

    @property
    def time_s(self) -> float:
        return sum(tc.time_s for tc in self.test_cases)


def render_junit_xml(
    suites: list[JUnitTestSuite],
    *,
    encoding: str = "unicode",
) -> str:
    """Render test suites as JUnit XML string.

    Args:
        suites: List of test suites to render.
        encoding: XML encoding. Use "unicode" for a string result.

    Returns:
        JUnit XML as a string.
    """
    root = Element("testsuites")

    total_tests = sum(s.tests for s in suites)
    total_failures = sum(s.failures for s in suites)
    total_errors = sum(s.errors for s in suites)
    total_time = sum(s.time_s for s in suites)

    root.set("tests", str(total_tests))
    root.set("failures", str(total_failures))
    root.set("errors", str(total_errors))
    root.set("time", f"{total_time:.3f}")

    for suite in suites:
        _render_suite(root, suite)

    xml_bytes = tostring(root, encoding=encoding, xml_declaration=True)
    if isinstance(xml_bytes, bytes):
        return xml_bytes.decode(encoding)
    return xml_bytes


def _render_suite(parent: Element, suite: JUnitTestSuite) -> None:
    """Render a single test suite element."""
    el = SubElement(parent, "testsuite")
    el.set("name", suite.name)
    el.set("tests", str(suite.tests))
    el.set("failures", str(suite.failures))
    el.set("errors", str(suite.errors))
    el.set("skipped", str(suite.skipped))
    el.set("time", f"{suite.time_s:.3f}")

    if suite.timestamp:
        el.set("timestamp", suite.timestamp)
    if suite.hostname:
        el.set("hostname", suite.hostname)

    if suite.properties:
        props_el = SubElement(el, "properties")
        for prop in suite.properties:
            p = SubElement(props_el, "property")
            p.set("name", prop.name)
            p.set("value", prop.value)

    for tc in suite.test_cases:
        _render_test_case(el, tc)


def _render_test_case(parent: Element, tc: JUnitTestCase) -> None:
    """Render a single test case element."""
    el = SubElement(parent, "testcase")
    el.set("name", tc.name)
    el.set("classname", tc.classname)
    el.set("time", f"{tc.time_s:.3f}")

    if tc.properties:
        props_el = SubElement(el, "properties")
        for prop in tc.properties:
            p = SubElement(props_el, "property")
            p.set("name", prop.name)
            p.set("value", prop.value)

    if tc.is_failure:
        fail_el = SubElement(el, "failure")
        fail_el.set("message", tc.failure_message)
        if tc.failure_text:
            fail_el.text = tc.failure_text

    if tc.is_error:
        err_el = SubElement(el, "error")
        err_el.set("message", tc.error_message)
        if tc.error_text:
            err_el.text = tc.error_text

    if tc.is_skipped:
        skip_el = SubElement(el, "skipped")
        skip_el.set("message", tc.skipped_message)

    if tc.stdout:
        stdout_el = SubElement(el, "system-out")
        stdout_el.text = tc.stdout

    if tc.stderr:
        stderr_el = SubElement(el, "system-err")
        stderr_el.text = tc.stderr


def from_run_summary(
    summary: "RunSummary",
    *,
    suite_name: str = "checkagent",
    test_details: list[dict[str, str]] | None = None,
) -> JUnitTestSuite:
    """Convert a RunSummary into a JUnitTestSuite.

    If test_details is provided, each entry should have:
        - name: test name
        - classname: test class/module
        - time_s: duration (optional, defaults to 0)
        - status: "passed", "failed", "error", or "skipped"
        - message: failure/error/skip message (optional)
        - text: failure/error detail text (optional)

    If test_details is not provided, synthetic test cases are created
    from the summary counts.
    """
    from checkagent.ci.reporter import RunSummary as _RS  # noqa: F811

    cases: list[JUnitTestCase] = []

    if test_details:
        for detail in test_details:
            tc = JUnitTestCase(
                name=detail["name"],
                classname=detail.get("classname", suite_name),
                time_s=float(detail.get("time_s", 0)),
            )
            status = detail.get("status", "passed")
            msg = detail.get("message", "")
            text = detail.get("text", "")

            if status == "failed":
                tc.failure_message = msg or "Test failed"
                tc.failure_text = text or None
            elif status == "error":
                tc.error_message = msg or "Test error"
                tc.error_text = text or None
            elif status == "skipped":
                tc.skipped_message = msg or "Test skipped"

            cases.append(tc)
    else:
        # Synthesize from counts
        avg_time = summary.duration_s / max(summary.total, 1)
        for i in range(summary.passed):
            cases.append(JUnitTestCase(
                name=f"test_{i + 1}",
                classname=suite_name,
                time_s=avg_time,
            ))
        for i in range(summary.failed):
            name = summary.regressions[i] if i < len(summary.regressions) else f"test_failed_{i + 1}"
            cases.append(JUnitTestCase(
                name=name,
                classname=suite_name,
                time_s=avg_time,
                failure_message="Test failed",
            ))
        for i in range(summary.skipped):
            cases.append(JUnitTestCase(
                name=f"test_skipped_{i + 1}",
                classname=suite_name,
                time_s=0.0,
                skipped_message="Test skipped",
            ))
        for i in range(summary.errors):
            cases.append(JUnitTestCase(
                name=f"test_error_{i + 1}",
                classname=suite_name,
                time_s=avg_time,
                error_message="Test error",
            ))

    return JUnitTestSuite(name=suite_name, test_cases=cases)


def from_quality_gate_report(
    report: "QualityGateReport",
    *,
    suite_name: str = "checkagent.quality_gates",
) -> JUnitTestSuite:
    """Convert a QualityGateReport into a JUnitTestSuite.

    Each gate becomes a test case. Blocked gates become failures,
    warned gates become test cases with properties noting the warning,
    and skipped gates become skipped test cases.
    """
    from checkagent.ci.quality_gate import GateVerdict

    cases: list[JUnitTestCase] = []

    for result in report.results:
        props = []
        if result.actual is not None:
            props.append(JUnitProperty("actual", f"{result.actual:.4f}"))
        if result.threshold is not None:
            props.append(JUnitProperty("threshold", f"{result.threshold:.4f}"))
        props.append(JUnitProperty("direction", result.direction))

        tc = JUnitTestCase(
            name=f"gate_{result.metric}",
            classname=suite_name,
            properties=props,
        )

        if result.verdict == GateVerdict.BLOCKED:
            tc.failure_message = result.message or f"Gate '{result.metric}' blocked"
        elif result.verdict == GateVerdict.SKIPPED:
            tc.skipped_message = result.message or f"Gate '{result.metric}' skipped"
        elif result.verdict == GateVerdict.WARNED:
            tc.properties.append(JUnitProperty("warning", result.message))

        cases.append(tc)

    return JUnitTestSuite(name=suite_name, test_cases=cases)
