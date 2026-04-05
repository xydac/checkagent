"""Tests for ToolCallBoundaryValidator."""

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.safety.taxonomy import SafetyCategory, Severity
from checkagent.safety.tool_boundary import (
    ToolBoundary,
    ToolCallBoundaryValidator,
)


def _make_run(*tool_calls: ToolCall) -> AgentRun:
    """Helper to create an AgentRun with the given tool calls."""
    return AgentRun(
        input=AgentInput(query="test"),
        steps=[Step(tool_calls=list(tool_calls))],
        final_output="done",
    )


# ---------------------------------------------------------------------------
# Allowed tools (allowlist)
# ---------------------------------------------------------------------------


class TestAllowedTools:
    def test_allowed_tool_passes(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_tools={"search", "read_file"})
        )
        run = _make_run(ToolCall(name="search", arguments={"q": "hello"}))
        result = validator.evaluate_run(run)
        assert result.passed
        assert result.finding_count == 0

    def test_unauthorized_tool_fails(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_tools={"search"})
        )
        run = _make_run(ToolCall(name="delete_file", arguments={"path": "/etc/passwd"}))
        result = validator.evaluate_run(run)
        assert not result.passed
        assert result.finding_count == 1
        assert result.findings[0].severity == Severity.HIGH
        assert "delete_file" in result.findings[0].description

    def test_empty_allowlist_permits_all(self):
        validator = ToolCallBoundaryValidator(ToolBoundary())
        run = _make_run(ToolCall(name="anything", arguments={}))
        result = validator.evaluate_run(run)
        assert result.passed

    def test_multiple_unauthorized_tools(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_tools={"search"})
        )
        run = _make_run(
            ToolCall(name="delete_file", arguments={}),
            ToolCall(name="exec_command", arguments={}),
        )
        result = validator.evaluate_run(run)
        assert not result.passed
        assert result.finding_count == 2


# ---------------------------------------------------------------------------
# Forbidden tools (blocklist)
# ---------------------------------------------------------------------------


class TestForbiddenTools:
    def test_forbidden_tool_fails(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_tools={"rm", "drop_table"})
        )
        run = _make_run(ToolCall(name="rm", arguments={"path": "/tmp/x"}))
        result = validator.evaluate_run(run)
        assert not result.passed
        assert result.findings[0].severity == Severity.CRITICAL
        assert "Forbidden" in result.findings[0].description

    def test_non_forbidden_tool_passes(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_tools={"rm"})
        )
        run = _make_run(ToolCall(name="ls", arguments={}))
        result = validator.evaluate_run(run)
        assert result.passed

    def test_both_forbidden_and_allowlist(self):
        """A tool on the forbidden list AND not on the allowlist gets two findings."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_tools={"search"}, forbidden_tools={"exec"})
        )
        run = _make_run(ToolCall(name="exec", arguments={}))
        result = validator.evaluate_run(run)
        assert not result.passed
        # One CRITICAL (forbidden) + one HIGH (unauthorized)
        assert result.finding_count == 2


# ---------------------------------------------------------------------------
# Path boundaries
# ---------------------------------------------------------------------------


class TestPathBoundaries:
    def test_path_within_boundary_passes(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/home/user/project"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/home/user/project/src/main.py"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_path_outside_boundary_fails(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/home/user/project"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/etc/passwd"})
        )
        result = validator.evaluate_run(run)
        assert not result.passed
        assert "/etc/passwd" in result.findings[0].evidence

    def test_multiple_allowed_paths(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/home/user/project", "/tmp"])
        )
        run = _make_run(
            ToolCall(name="write_file", arguments={"file": "/tmp/output.txt"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_no_path_checking_when_empty(self):
        validator = ToolCallBoundaryValidator(ToolBoundary())
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/etc/shadow"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_non_path_args_ignored(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/home/user"])
        )
        run = _make_run(
            ToolCall(name="search", arguments={"query": "/etc/passwd"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_various_path_arg_names(self):
        """All recognized path argument names are checked."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/safe"])
        )
        for arg_name in ["path", "file", "filepath", "file_path", "filename", "directory", "dir"]:
            run = _make_run(
                ToolCall(name="tool", arguments={arg_name: "/unsafe/data"})
            )
            result = validator.evaluate_run(run)
            assert not result.passed, f"Failed for arg name: {arg_name}"


# ---------------------------------------------------------------------------
# Forbidden argument patterns
# ---------------------------------------------------------------------------


class TestForbiddenArgumentPatterns:
    def test_forbidden_pattern_detected(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_argument_patterns={"command": r"rm\s+-rf"})
        )
        run = _make_run(
            ToolCall(name="exec", arguments={"command": "rm -rf /important"})
        )
        result = validator.evaluate_run(run)
        assert not result.passed
        assert "rm -rf" in result.findings[0].evidence

    def test_safe_pattern_passes(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_argument_patterns={"command": r"rm\s+-rf"})
        )
        run = _make_run(
            ToolCall(name="exec", arguments={"command": "ls -la"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_pattern_only_matches_named_arg(self):
        """Pattern for 'command' should not match 'description'."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_argument_patterns={"command": r"drop table"})
        )
        run = _make_run(
            ToolCall(name="note", arguments={"description": "drop table students"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_multiple_patterns(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_argument_patterns={
                "command": r"rm\s+-rf",
                "sql": r"(?i)drop\s+table",
            })
        )
        run = _make_run(
            ToolCall(name="db", arguments={"sql": "DROP TABLE users"})
        )
        result = validator.evaluate_run(run)
        assert not result.passed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_tool_calls_passes(self):
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_tools={"search"})
        )
        run = AgentRun(
            input=AgentInput(query="test"),
            steps=[Step()],
            final_output="no tools used",
        )
        result = validator.evaluate_run(run)
        assert result.passed
        assert result.details["tool_calls_checked"] == 0

    def test_text_evaluate_always_passes(self):
        """Text-only evaluate is a no-op for tool boundary checks."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_tools={"everything"})
        )
        result = validator.evaluate("some text")
        assert result.passed

    def test_category_is_tool_misuse(self):
        validator = ToolCallBoundaryValidator()
        assert validator.category == SafetyCategory.TOOL_MISUSE

    def test_default_boundary_permits_everything(self):
        validator = ToolCallBoundaryValidator()
        run = _make_run(
            ToolCall(name="anything", arguments={"path": "/anywhere", "command": "rm -rf /"})
        )
        result = validator.evaluate_run(run)
        assert result.passed
