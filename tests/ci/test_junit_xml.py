"""Tests for JUnit XML output (F5.6)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from checkagent.ci.junit_xml import (
    JUnitProperty,
    JUnitTestCase,
    JUnitTestSuite,
    from_quality_gate_report,
    from_run_summary,
    render_junit_xml,
)
from checkagent.ci.quality_gate import GateResult, GateVerdict, QualityGateReport
from checkagent.ci.reporter import RunSummary


class TestJUnitTestCase:
    """Tests for JUnitTestCase data class."""

    def test_passed_by_default(self):
        tc = JUnitTestCase(name="test_x", classname="mod")
        assert tc.is_passed
        assert not tc.is_failure
        assert not tc.is_error
        assert not tc.is_skipped

    def test_failure(self):
        tc = JUnitTestCase(name="test_x", classname="mod", failure_message="oops")
        assert tc.is_failure
        assert not tc.is_passed

    def test_error(self):
        tc = JUnitTestCase(name="test_x", classname="mod", error_message="boom")
        assert tc.is_error
        assert not tc.is_passed

    def test_skipped(self):
        tc = JUnitTestCase(name="test_x", classname="mod", skipped_message="skip")
        assert tc.is_skipped
        assert not tc.is_passed


class TestJUnitTestSuite:
    """Tests for JUnitTestSuite computed properties."""

    def test_empty_suite(self):
        suite = JUnitTestSuite(name="empty")
        assert suite.tests == 0
        assert suite.failures == 0
        assert suite.errors == 0
        assert suite.skipped == 0
        assert suite.time_s == 0.0

    def test_counts(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(name="a", classname="m"),
            JUnitTestCase(name="b", classname="m", failure_message="f"),
            JUnitTestCase(name="c", classname="m", error_message="e"),
            JUnitTestCase(name="d", classname="m", skipped_message="s"),
        ])
        assert suite.tests == 4
        assert suite.failures == 1
        assert suite.errors == 1
        assert suite.skipped == 1

    def test_time_aggregation(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(name="a", classname="m", time_s=1.5),
            JUnitTestCase(name="b", classname="m", time_s=2.3),
        ])
        assert abs(suite.time_s - 3.8) < 0.001


class TestRenderJUnitXML:
    """Tests for XML rendering."""

    def test_empty_suites(self):
        xml = render_junit_xml([])
        root = ET.fromstring(xml)
        assert root.tag == "testsuites"
        assert root.get("tests") == "0"

    def test_xml_declaration(self):
        xml = render_junit_xml([])
        assert xml.startswith("<?xml")

    def test_single_passing_test(self):
        suite = JUnitTestSuite(name="my_tests", test_cases=[
            JUnitTestCase(name="test_one", classname="tests.test_foo", time_s=0.123),
        ])
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)

        assert root.get("tests") == "1"
        assert root.get("failures") == "0"

        ts = root.find("testsuite")
        assert ts is not None
        assert ts.get("name") == "my_tests"

        tc = ts.find("testcase")
        assert tc is not None
        assert tc.get("name") == "test_one"
        assert tc.get("classname") == "tests.test_foo"
        assert tc.get("time") == "0.123"
        assert tc.find("failure") is None

    def test_failure_element(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(
                name="test_fail",
                classname="m",
                failure_message="assertion error",
                failure_text="assert 1 == 2",
            ),
        ])
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        tc = root.find(".//testcase")
        fail = tc.find("failure")
        assert fail is not None
        assert fail.get("message") == "assertion error"
        assert fail.text == "assert 1 == 2"

    def test_error_element(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(
                name="test_err",
                classname="m",
                error_message="runtime error",
                error_text="traceback...",
            ),
        ])
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        tc = root.find(".//testcase")
        err = tc.find("error")
        assert err is not None
        assert err.get("message") == "runtime error"
        assert err.text == "traceback..."

    def test_skipped_element(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(name="test_skip", classname="m", skipped_message="not ready"),
        ])
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        tc = root.find(".//testcase")
        skip = tc.find("skipped")
        assert skip is not None
        assert skip.get("message") == "not ready"

    def test_stdout_stderr(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(
                name="test_io",
                classname="m",
                stdout="hello stdout",
                stderr="hello stderr",
            ),
        ])
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        tc = root.find(".//testcase")
        assert tc.find("system-out").text == "hello stdout"
        assert tc.find("system-err").text == "hello stderr"

    def test_properties(self):
        suite = JUnitTestSuite(
            name="s",
            properties=[JUnitProperty("framework", "checkagent")],
            test_cases=[
                JUnitTestCase(
                    name="test_p",
                    classname="m",
                    properties=[JUnitProperty("layer", "mock")],
                ),
            ],
        )
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)

        # Suite-level properties
        ts = root.find("testsuite")
        suite_props = ts.find("properties")
        assert suite_props is not None
        prop = suite_props.find("property")
        assert prop.get("name") == "framework"
        assert prop.get("value") == "checkagent"

        # Test-case-level properties
        tc = ts.find("testcase")
        tc_props = tc.find("properties")
        assert tc_props is not None
        prop = tc_props.find("property")
        assert prop.get("name") == "layer"
        assert prop.get("value") == "mock"

    def test_multiple_suites(self):
        suites = [
            JUnitTestSuite(name="suite_a", test_cases=[
                JUnitTestCase(name="t1", classname="a", time_s=0.1),
            ]),
            JUnitTestSuite(name="suite_b", test_cases=[
                JUnitTestCase(name="t2", classname="b", time_s=0.2),
                JUnitTestCase(name="t3", classname="b", time_s=0.3, failure_message="f"),
            ]),
        ]
        xml = render_junit_xml(suites)
        root = ET.fromstring(xml)
        assert root.get("tests") == "3"
        assert root.get("failures") == "1"
        assert root.get("errors") == "0"
        assert len(root.findall("testsuite")) == 2

    def test_timestamp_and_hostname(self):
        suite = JUnitTestSuite(
            name="s",
            timestamp="2026-04-05T15:30:00",
            hostname="ci-runner-1",
        )
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        ts = root.find("testsuite")
        assert ts.get("timestamp") == "2026-04-05T15:30:00"
        assert ts.get("hostname") == "ci-runner-1"

    def test_valid_xml_with_special_chars(self):
        suite = JUnitTestSuite(name="s", test_cases=[
            JUnitTestCase(
                name="test_special",
                classname="m",
                failure_message='error: x < 5 & y > 3 "quoted"',
                stdout="output with <angle> & ampersand",
            ),
        ])
        xml = render_junit_xml([suite])
        # Should parse without errors
        root = ET.fromstring(xml)
        tc = root.find(".//testcase")
        assert tc.find("failure").get("message") == 'error: x < 5 & y > 3 "quoted"'


class TestFromRunSummary:
    """Tests for converting RunSummary to JUnit XML."""

    def test_all_passed(self):
        summary = RunSummary(total=3, passed=3, duration_s=1.5)
        suite = from_run_summary(summary)
        assert suite.tests == 3
        assert suite.failures == 0
        assert all(tc.is_passed for tc in suite.test_cases)

    def test_mixed_results(self):
        summary = RunSummary(total=6, passed=3, failed=1, skipped=1, errors=1, duration_s=3.0)
        suite = from_run_summary(summary)
        assert suite.tests == 6
        assert suite.failures == 1
        assert suite.errors == 1
        assert suite.skipped == 1

    def test_regressions_used_as_names(self):
        summary = RunSummary(
            total=2, passed=0, failed=2,
            regressions=["test_login", "test_signup"],
            duration_s=1.0,
        )
        suite = from_run_summary(summary)
        names = [tc.name for tc in suite.test_cases if tc.is_failure]
        assert "test_login" in names
        assert "test_signup" in names

    def test_custom_suite_name(self):
        summary = RunSummary(total=1, passed=1, duration_s=0.1)
        suite = from_run_summary(summary, suite_name="my_agent")
        assert suite.name == "my_agent"

    def test_with_test_details(self):
        summary = RunSummary(total=2, passed=1, failed=1, duration_s=0.5)
        details = [
            {"name": "test_a", "classname": "tests.mod", "status": "passed", "time_s": "0.1"},
            {"name": "test_b", "classname": "tests.mod", "status": "failed", "message": "bad"},
        ]
        suite = from_run_summary(summary, test_details=details)
        assert suite.tests == 2
        passed = [tc for tc in suite.test_cases if tc.is_passed]
        failed = [tc for tc in suite.test_cases if tc.is_failure]
        assert len(passed) == 1
        assert len(failed) == 1
        assert failed[0].failure_message == "bad"

    def test_zero_total(self):
        summary = RunSummary(total=0, passed=0, duration_s=0.0)
        suite = from_run_summary(summary)
        assert suite.tests == 0

    def test_renders_valid_xml(self):
        summary = RunSummary(total=5, passed=3, failed=1, skipped=1, duration_s=2.5)
        suite = from_run_summary(summary)
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        assert root.get("tests") == "5"


class TestFromQualityGateReport:
    """Tests for converting QualityGateReport to JUnit XML."""

    def test_all_passed_gates(self):
        report = QualityGateReport(results=[
            GateResult(metric="accuracy", verdict=GateVerdict.PASSED, actual=0.95, threshold=0.90),
            GateResult(metric="speed", verdict=GateVerdict.PASSED, actual=0.85, threshold=0.80),
        ])
        suite = from_quality_gate_report(report)
        assert suite.tests == 2
        assert suite.failures == 0
        assert all(tc.is_passed for tc in suite.test_cases)

    def test_blocked_gate_becomes_failure(self):
        report = QualityGateReport(results=[
            GateResult(
                metric="accuracy",
                verdict=GateVerdict.BLOCKED,
                actual=0.50,
                threshold=0.90,
                message="accuracy too low",
            ),
        ])
        suite = from_quality_gate_report(report)
        assert suite.failures == 1
        tc = suite.test_cases[0]
        assert tc.is_failure
        assert "accuracy too low" in tc.failure_message

    def test_warned_gate_passes_with_property(self):
        report = QualityGateReport(results=[
            GateResult(
                metric="cost",
                verdict=GateVerdict.WARNED,
                actual=0.60,
                threshold=0.50,
                message="cost slightly high",
            ),
        ])
        suite = from_quality_gate_report(report)
        assert suite.failures == 0
        tc = suite.test_cases[0]
        assert tc.is_passed  # Warnings don't fail
        warning_props = [p for p in tc.properties if p.name == "warning"]
        assert len(warning_props) == 1
        assert "cost slightly high" in warning_props[0].value

    def test_skipped_gate(self):
        report = QualityGateReport(results=[
            GateResult(metric="latency", verdict=GateVerdict.SKIPPED, message="not measured"),
        ])
        suite = from_quality_gate_report(report)
        assert suite.skipped == 1
        assert suite.test_cases[0].is_skipped

    def test_custom_suite_name(self):
        report = QualityGateReport(results=[])
        suite = from_quality_gate_report(report, suite_name="gates.custom")
        assert suite.name == "gates.custom"

    def test_properties_on_gate_cases(self):
        report = QualityGateReport(results=[
            GateResult(
                metric="acc", verdict=GateVerdict.PASSED,
                actual=0.95, threshold=0.90, direction="min",
            ),
        ])
        suite = from_quality_gate_report(report)
        tc = suite.test_cases[0]
        prop_names = [p.name for p in tc.properties]
        assert "actual" in prop_names
        assert "threshold" in prop_names
        assert "direction" in prop_names

    def test_renders_valid_xml(self):
        report = QualityGateReport(results=[
            GateResult(
                metric="a", verdict=GateVerdict.PASSED,
                actual=0.9, threshold=0.8,
            ),
            GateResult(
                metric="b", verdict=GateVerdict.BLOCKED,
                actual=0.3, threshold=0.5, message="low",
            ),
        ])
        suite = from_quality_gate_report(report)
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"
