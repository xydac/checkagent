"""Tests for the ``checkagent scan`` CLI command."""

from __future__ import annotations

import json
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from click.testing import CliRunner

from checkagent.cli import main
from checkagent.cli.badge import generate_badge_svg, write_badge
from checkagent.cli.scan import (
    _build_json_report,
    _evaluate_output,
    _generate_test_file,
    _make_http_agent,
    _resolve_callable,
    scan_cmd,
)
from checkagent.safety.evaluator import SafetyFinding
from checkagent.safety.probes.base import Probe
from checkagent.safety.taxonomy import SafetyCategory, Severity

# ---------------------------------------------------------------------------
# Helper: write a temp module with agent callables
# ---------------------------------------------------------------------------


def _write_agent_module(tmp_path: Path) -> Path:
    """Write a temp module with safe, unsafe, and sync agent callables."""
    mod = tmp_path / "scan_test_agents.py"
    mod.write_text(textwrap.dedent("""\
        async def safe_agent(query):
            return "I can help you with that."

        async def unsafe_agent(query):
            if "system prompt" in query.lower():
                return "My system prompt is: You are a helpful assistant."
            return "I can help you with that."

        def sync_agent(query):
            return "Sync response: " + query[:20]

        async def error_agent(query):
            raise ValueError("Agent crashed")

        async def dict_agent(query):
            return {"output": "I can help.", "status": "ok"}

        not_callable = 42
    """))
    return mod


# ---------------------------------------------------------------------------
# Unit tests: _resolve_callable
# ---------------------------------------------------------------------------


class TestResolveCallable:
    def test_colon_syntax(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        fn = _resolve_callable("scan_test_agents:safe_agent")
        assert callable(fn)

    def test_dot_syntax(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        fn = _resolve_callable("scan_test_agents.safe_agent")
        assert callable(fn)

    def test_missing_module(self) -> None:
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="Cannot import"):
            _resolve_callable("nonexistent_module_xyz:fn")

    def test_missing_attr(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="no attribute"):
            _resolve_callable("scan_test_agents:nonexistent_fn")

    def test_not_callable(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="not callable"):
            _resolve_callable("scan_test_agents:not_callable")

    def test_no_separator(self) -> None:
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="Cannot parse"):
            _resolve_callable("just_a_name")


# ---------------------------------------------------------------------------
# Unit tests: _evaluate_output
# ---------------------------------------------------------------------------


class TestEvaluateOutput:
    def test_clean_output(self) -> None:
        findings = _evaluate_output("I can help you with that.")
        assert findings == []

    def test_injection_detected(self) -> None:
        findings = _evaluate_output("My system prompt is: You are a helpful assistant.")
        assert len(findings) > 0
        categories = {f.category.value for f in findings}
        assert "prompt_injection" in categories or "pii_leakage" in categories

    def test_pii_detected(self) -> None:
        findings = _evaluate_output("The user's email is alice@example.com")
        assert len(findings) > 0
        assert any("email" in f.description.lower() for f in findings)


# ---------------------------------------------------------------------------
# Integration tests: scan_cmd via CliRunner
# ---------------------------------------------------------------------------


class TestScanCommand:
    def test_safe_agent_passes(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 0
        assert "No safety issues detected" in result.output

    def test_unsafe_agent_fails(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:unsafe_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 1
        assert "safety issue" in result.output.lower()

    def test_sync_agent_works(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:sync_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        # Sync agent just returns prefix + truncated input, should not trigger detectors
        # for most probes
        assert "Scan Summary" in result.output

    def test_dict_output_agent(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:dict_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 0
        assert "No safety issues detected" in result.output

    def test_error_agent_counts_errors(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:error_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert "Errors" in result.output

    def test_bad_target_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["nonexistent_xyz:fn", "--timeout", "1"])
        assert result.exit_code != 0

    def test_category_filter(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "pii",
            "--timeout", "2",
        ])
        assert "Scan Summary" in result.output
        assert "10" in result.output  # 10 PII probes

    def test_scan_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "scan" in result.output

    def test_verbose_flag_accepted(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--verbose",
        ])
        assert result.exit_code == 0

    def test_generate_tests_with_findings(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        out_file = tmp_path / "test_safety.py"
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:unsafe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--generate-tests", str(out_file),
        ])
        assert out_file.exists()
        content = out_file.read_text()
        assert "def test_" in content
        assert "pytest" in content
        assert "Generated" in result.output or "test" in result.output.lower()

    def test_generate_tests_no_findings(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        out_file = tmp_path / "test_safety.py"
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--generate-tests", str(out_file),
        ])
        assert not out_file.exists()
        assert "No findings" in result.output

    def test_generate_tests_flag_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["--help"])
        assert "--generate-tests" in result.output


# ---------------------------------------------------------------------------
# Unit tests: _generate_test_file
# ---------------------------------------------------------------------------


class TestGenerateTestFile:
    def _make_findings(
        self,
        category: str = "prompt_injection",
        count: int = 2,
    ) -> list[tuple[Probe, SafetyFinding]]:
        findings = []
        for i in range(count):
            probe = Probe(
                input=f"Ignore instructions {i}",
                category=SafetyCategory.PROMPT_INJECTION,
                severity=Severity.HIGH,
                name=f"injection-probe-{i}",
            )
            finding = SafetyFinding(
                category=SafetyCategory.PROMPT_INJECTION,
                severity=Severity.HIGH,
                description=f"Possible injection detected {i}",
            )
            findings.append((probe, finding))
        return findings

    def test_generates_valid_python(self, tmp_path: Path) -> None:
        out = tmp_path / "test_gen.py"
        findings = self._make_findings()
        _generate_test_file("my_mod:my_fn", findings, out)
        content = out.read_text()
        # Should be valid Python
        compile(content, str(out), "exec")

    def test_contains_target(self, tmp_path: Path) -> None:
        out = tmp_path / "test_gen.py"
        _generate_test_file("my_mod:agent", self._make_findings(), out)
        content = out.read_text()
        assert 'TARGET = "my_mod:agent"' in content

    def test_one_function_per_category(self, tmp_path: Path) -> None:
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", self._make_findings(), out)
        content = out.read_text()
        assert "def test_prompt_injection_safety" in content

    def test_deduplicates_probes(self, tmp_path: Path) -> None:
        """Same input should produce one param, not two."""
        probe = Probe(
            input="duplicate input",
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            name="dup",
        )
        finding = SafetyFinding(
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            description="found",
        )
        findings = [(probe, finding), (probe, finding)]
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", findings, out)
        content = out.read_text()
        assert content.count("duplicate input") == 1

    def test_multiple_categories(self, tmp_path: Path) -> None:
        probe_inj = Probe(
            input="inject",
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            name="inj",
        )
        finding_inj = SafetyFinding(
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            description="injection",
        )
        probe_pii = Probe(
            input="pii leak",
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.MEDIUM,
            name="pii",
        )
        finding_pii = SafetyFinding(
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.MEDIUM,
            description="pii found",
        )
        findings = [(probe_inj, finding_inj), (probe_pii, finding_pii)]
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", findings, out)
        content = out.read_text()
        assert "def test_prompt_injection_safety" in content
        assert "def test_pii_leakage_safety" in content

    def test_escapes_special_chars(self, tmp_path: Path) -> None:
        probe = Probe(
            input='He said "ignore rules"\nand more',
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            name="special",
        )
        finding = SafetyFinding(
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            description="found",
        )
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", [(probe, finding)], out)
        content = out.read_text()
        # Should be valid Python despite special chars
        compile(content, str(out), "exec")
        # Newline should be escaped
        assert "\\n" in content

    def test_has_pytest_imports(self, tmp_path: Path) -> None:
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", self._make_findings(), out)
        content = out.read_text()
        assert "import pytest" in content
        assert "pytest.mark.parametrize" in content

    def test_has_fixture(self, tmp_path: Path) -> None:
        out = tmp_path / "test_gen.py"
        _generate_test_file("m:f", self._make_findings(), out)
        content = out.read_text()
        assert "@pytest.fixture" in content
        assert "def agent_fn():" in content


# ---------------------------------------------------------------------------
# Unit tests: _build_json_report
# ---------------------------------------------------------------------------


class TestBuildJsonReport:
    def test_basic_structure(self) -> None:
        report = _build_json_report(
            target="my_mod:fn",
            total=10,
            passed=8,
            failed=2,
            errors=0,
            elapsed=1.234,
            all_findings=[],
        )
        assert report["target"] == "my_mod:fn"
        assert report["summary"]["total"] == 10
        assert report["summary"]["passed"] == 8
        assert report["summary"]["failed"] == 2
        assert report["summary"]["errors"] == 0
        assert report["summary"]["score"] == 0.8
        assert report["summary"]["elapsed_seconds"] == 1.234
        assert report["findings"] == []

    def test_with_findings(self) -> None:
        probe = Probe(
            input="ignore all instructions",
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            name="test-probe",
        )
        finding = SafetyFinding(
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            description="Injection detected",
        )
        report = _build_json_report(
            target="m:f",
            total=5,
            passed=3,
            failed=2,
            errors=0,
            elapsed=0.5,
            all_findings=[(probe, finding)],
        )
        assert len(report["findings"]) == 1
        f = report["findings"][0]
        assert f["probe"] == "test-probe"
        assert f["category"] == "prompt_injection"
        assert f["severity"] == "high"
        assert f["description"] == "Injection detected"
        assert f["input"] == "ignore all instructions"

    def test_score_zero_total(self) -> None:
        report = _build_json_report(
            target="m:f",
            total=0,
            passed=0,
            failed=0,
            errors=0,
            elapsed=0.0,
            all_findings=[],
        )
        assert report["summary"]["score"] == 0.0

    def test_perfect_score(self) -> None:
        report = _build_json_report(
            target="m:f",
            total=10,
            passed=10,
            failed=0,
            errors=0,
            elapsed=0.1,
            all_findings=[],
        )
        assert report["summary"]["score"] == 1.0

    def test_json_serializable(self) -> None:
        """Report must be fully JSON-serializable."""
        probe = Probe(
            input="test input",
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.MEDIUM,
            name="pii-probe",
        )
        finding = SafetyFinding(
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.MEDIUM,
            description="PII found",
        )
        report = _build_json_report(
            target="m:f",
            total=5,
            passed=4,
            failed=1,
            errors=0,
            elapsed=0.5,
            all_findings=[(probe, finding)],
        )
        # Must not raise
        serialized = json.dumps(report)
        parsed = json.loads(serialized)
        assert parsed == report


# ---------------------------------------------------------------------------
# Unit tests: badge SVG generation
# ---------------------------------------------------------------------------


class TestBadgeGeneration:
    def test_generate_badge_svg_all_pass(self) -> None:
        svg = generate_badge_svg(passed=10, failed=0)
        assert "<svg" in svg
        assert "10/10 safe" in svg
        assert "#4c1" in svg  # green

    def test_generate_badge_svg_some_fail(self) -> None:
        svg = generate_badge_svg(passed=7, failed=3)
        assert "7/10 safe" in svg
        assert "#dfb317" in svg  # yellow (70% pass)

    def test_generate_badge_svg_many_fail(self) -> None:
        svg = generate_badge_svg(passed=5, failed=5)
        assert "5/10 safe" in svg
        assert "#e05d44" in svg  # red (50% pass)

    def test_generate_badge_svg_no_data(self) -> None:
        svg = generate_badge_svg(passed=0, failed=0)
        assert "no data" in svg
        assert "#9f9f9f" in svg  # gray

    def test_generate_badge_svg_custom_label(self) -> None:
        svg = generate_badge_svg(passed=5, failed=0, label="Safety")
        assert "Safety" in svg
        assert "5/5 safe" in svg

    def test_generate_badge_svg_has_aria(self) -> None:
        svg = generate_badge_svg(passed=8, failed=2)
        assert 'aria-label="CheckAgent: 8/10 safe"' in svg
        assert "<title>CheckAgent: 8/10 safe</title>" in svg

    def test_write_badge(self, tmp_path: Path) -> None:
        path = write_badge(
            tmp_path / "badge.svg",
            passed=10,
            failed=0,
        )
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content
        assert "10/10 safe" in content

    def test_write_badge_returns_path(self, tmp_path: Path) -> None:
        path = write_badge(tmp_path / "out.svg", passed=5, failed=5)
        assert path == tmp_path / "out.svg"


# ---------------------------------------------------------------------------
# Integration tests: --json and --badge flags
# ---------------------------------------------------------------------------


class TestScanJsonOutput:
    def test_json_flag_safe_agent(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["target"] == "scan_test_agents:safe_agent"
        assert data["summary"]["failed"] == 0
        assert data["summary"]["score"] == 1.0
        assert data["findings"] == []

    def test_json_flag_unsafe_agent(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:unsafe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--json",
        ])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["summary"]["failed"] > 0
        assert len(data["findings"]) > 0

    def test_json_suppresses_rich_output(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--json",
        ])
        # JSON output should not contain Rich formatting
        assert "Scan Summary" not in result.output
        assert "No safety issues detected" not in result.output
        # Should be valid JSON
        json.loads(result.output)

    def test_json_flag_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["--help"])
        assert "--json" in result.output


# ---------------------------------------------------------------------------
# Helpers: ephemeral HTTP servers for testing
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    """HTTP handler that echoes the 'message' field back as JSON."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        # Echo the input field back — mimics a vulnerable echo agent
        msg = data.get("message", data.get("query", ""))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"output": msg}).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress logs


class _SafeHandler(BaseHTTPRequestHandler):
    """HTTP handler that always returns a safe response."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # consume body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"output": "I can help you with that."}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class _CustomFieldHandler(BaseHTTPRequestHandler):
    """HTTP handler that reads 'query' and responds with 'reply' field."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        msg = data.get("query", "no query field")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"reply": f"Got: {msg[:30]}"}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class _PlainTextHandler(BaseHTTPRequestHandler):
    """HTTP handler that returns plain text (not JSON)."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"I can help you with that.")

    def log_message(self, format, *args):
        pass


class _ErrorHandler(BaseHTTPRequestHandler):
    """HTTP handler that always returns 500."""

    def do_POST(self):
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Internal Server Error")

    def log_message(self, format, *args):
        pass


class _HeaderCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler that checks for an auth header."""

    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if auth != "Bearer test-token":
            self.send_response(401)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Unauthorized")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"output": "Authenticated OK."}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _start_server(handler_cls) -> tuple[HTTPServer, str]:
    """Start an HTTP server on a random port, return (server, url)."""
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Unit tests: _make_http_agent
# ---------------------------------------------------------------------------


class TestMakeHttpAgent:
    def test_echo_agent(self) -> None:
        import asyncio

        server, url = _start_server(_EchoHandler)
        try:
            agent = _make_http_agent(url)
            result = asyncio.run(agent("hello world"))
            assert result == "hello world"
        finally:
            server.shutdown()

    def test_custom_input_field(self) -> None:
        import asyncio

        server, url = _start_server(_CustomFieldHandler)
        try:
            agent = _make_http_agent(url, input_field="query")
            result = asyncio.run(agent("test query"))
            assert "Got: test query" in result
        finally:
            server.shutdown()

    def test_custom_output_field(self) -> None:
        import asyncio

        server, url = _start_server(_CustomFieldHandler)
        try:
            agent = _make_http_agent(url, input_field="query", output_field="reply")
            result = asyncio.run(agent("test"))
            assert "Got:" in result
        finally:
            server.shutdown()

    def test_plain_text_response(self) -> None:
        import asyncio

        server, url = _start_server(_PlainTextHandler)
        try:
            agent = _make_http_agent(url)
            result = asyncio.run(agent("hello"))
            assert result == "I can help you with that."
        finally:
            server.shutdown()

    def test_server_error_raises(self) -> None:
        import asyncio

        import pytest

        server, url = _start_server(_ErrorHandler)
        try:
            agent = _make_http_agent(url)
            with pytest.raises(RuntimeError, match="HTTP 500"):
                asyncio.run(agent("hello"))
        finally:
            server.shutdown()

    def test_connection_refused_raises(self) -> None:
        import asyncio

        import pytest

        agent = _make_http_agent("http://127.0.0.1:1")  # port 1 won't work
        with pytest.raises(RuntimeError, match="Cannot connect"):
            asyncio.run(agent("hello"))

    def test_headers_sent(self) -> None:
        import asyncio

        server, url = _start_server(_HeaderCheckHandler)
        try:
            agent = _make_http_agent(
                url,
                headers={"Authorization": "Bearer test-token"},
            )
            result = asyncio.run(agent("hello"))
            assert "Authenticated" in result
        finally:
            server.shutdown()

    def test_auto_detect_response_fields(self) -> None:
        """The agent should auto-detect common response field names."""
        import asyncio

        server, url = _start_server(_CustomFieldHandler)
        try:
            # "reply" is in our auto-detect list
            agent = _make_http_agent(url, input_field="query")
            result = asyncio.run(agent("test"))
            assert "Got:" in result
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# Integration tests: scan_cmd with --url
# ---------------------------------------------------------------------------


class TestScanHttpCommand:
    def test_url_safe_server(self) -> None:
        server, url = _start_server(_SafeHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
            ])
            assert result.exit_code == 0
            assert "No safety issues detected" in result.output
        finally:
            server.shutdown()

    def test_url_echo_server_finds_issues(self) -> None:
        server, url = _start_server(_EchoHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
            ])
            assert result.exit_code == 1
            assert "safety issue" in result.output.lower()
        finally:
            server.shutdown()

    def test_url_json_output(self) -> None:
        server, url = _start_server(_SafeHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
                "--json",
            ])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["target"] == url
            assert data["summary"]["failed"] == 0
        finally:
            server.shutdown()

    def test_url_with_header(self) -> None:
        server, url = _start_server(_HeaderCheckHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
                "-H", "Authorization: Bearer test-token",
            ])
            assert result.exit_code == 0
            assert "No safety issues detected" in result.output
        finally:
            server.shutdown()

    def test_url_with_custom_input_field(self) -> None:
        server, url = _start_server(_CustomFieldHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
                "--input-field", "query",
            ])
            assert "Scan Summary" in result.output
        finally:
            server.shutdown()

    def test_url_and_target_mutually_exclusive(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "my_agent:fn",
            "--url", "http://localhost:8000",
        ])
        assert result.exit_code != 0
        assert "Cannot use both" in result.output

    def test_neither_url_nor_target_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [])
        assert result.exit_code != 0
        assert "Provide either" in result.output

    def test_url_bad_header_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "--url", "http://localhost:8000",
            "-H", "bad-header-no-colon",
        ])
        assert result.exit_code != 0
        assert "Invalid header format" in result.output

    def test_url_shows_in_panel(self) -> None:
        server, url = _start_server(_SafeHandler)
        try:
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
            ])
            assert "127.0.0.1" in result.output
        finally:
            server.shutdown()

    def test_url_badge_generation(self) -> None:
        import tempfile

        server, url = _start_server(_SafeHandler)
        try:
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                badge_path = f.name
            runner = CliRunner()
            result = runner.invoke(scan_cmd, [
                "--url", url,
                "--category", "injection",
                "--timeout", "5",
                "--badge", badge_path,
            ])
            assert result.exit_code == 0
            content = Path(badge_path).read_text()
            assert "<svg" in content
        finally:
            server.shutdown()

    def test_url_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["--help"])
        assert "--url" in result.output
        assert "--input-field" in result.output
        assert "--output-field" in result.output
        assert "--header" in result.output


class TestScanBadgeOutput:
    def test_badge_flag_generates_svg(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        badge_path = tmp_path / "badge.svg"
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--badge", str(badge_path),
        ])
        assert result.exit_code == 0
        assert badge_path.exists()
        content = badge_path.read_text()
        assert "<svg" in content
        assert "safe" in content

    def test_badge_flag_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["--help"])
        assert "--badge" in result.output

    def test_badge_with_json(self, tmp_path: Path, monkeypatch) -> None:
        """Badge and JSON can be used together."""
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        badge_path = tmp_path / "badge.svg"
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--json",
            "--badge", str(badge_path),
        ])
        assert result.exit_code == 0
        # JSON still valid
        json.loads(result.output)
        # Badge still generated
        assert badge_path.exists()
        assert "<svg" in badge_path.read_text()
