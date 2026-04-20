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

    def test_path_prefix_confusion_blocked(self):
        """F-024: /dataextra must not match allowed path /data."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/dataextra/file.txt"})
        )
        result = validator.evaluate_run(run)
        assert not result.passed

    def test_path_exact_match_passes(self):
        """Exact path equal to allowed prefix should pass."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/data"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_path_traversal_blocked(self):
        """F-025: ../traversal must be normalized before boundary check."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/data/../etc/passwd"})
        )
        result = validator.evaluate_run(run)
        assert not result.passed

    def test_path_traversal_within_boundary_passes(self):
        """Traversal that stays within the boundary should pass."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/data/sub/../other/file.txt"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_path_dot_segments_normalized(self):
        """/data/./sub should normalize to /data/sub and pass."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/data/./sub/file.txt"})
        )
        result = validator.evaluate_run(run)
        assert result.passed

    def test_allowed_path_with_trailing_slash(self):
        """Trailing slash on allowed path should not break matching."""
        validator = ToolCallBoundaryValidator(
            ToolBoundary(allowed_paths=["/data/"])
        )
        run = _make_run(
            ToolCall(name="read_file", arguments={"path": "/data/file.txt"})
        )
        result = validator.evaluate_run(run)
        assert result.passed


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

    def test_text_evaluate_raises_not_implemented(self):
        """evaluate(text) raises because tool boundary needs structured data."""
        import pytest

        validator = ToolCallBoundaryValidator(
            ToolBoundary(forbidden_tools={"everything"})
        )
        with pytest.raises(NotImplementedError, match="evaluate_run"):
            validator.evaluate("some text")

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


class TestF109LegacyKwargsCompat:
    """F-109 regression: ToolCallBoundaryValidator accepts old kwargs with DeprecationWarning."""

    def test_legacy_allowed_tools_set_emits_deprecation(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = ToolCallBoundaryValidator(allowed_tools={"search", "read_file"})
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ToolBoundary" in str(w[0].message)
        assert v.boundary.allowed_tools == {"search", "read_file"}

    def test_legacy_forbidden_tools_list_coerced_to_set(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = ToolCallBoundaryValidator(forbidden_tools=["delete", "drop_table"])
        assert len(w) == 1
        assert v.boundary.forbidden_tools == {"delete", "drop_table"}

    def test_legacy_allowed_paths(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = ToolCallBoundaryValidator(allowed_paths=["/data"])
        assert len(w) == 1
        assert v.boundary.allowed_paths == ["/data"]

    def test_legacy_forbidden_argument_patterns(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = ToolCallBoundaryValidator(
                forbidden_argument_patterns={"command": r"rm\s+-rf"}
            )
        assert len(w) == 1
        assert "command" in v.boundary.forbidden_argument_patterns

    def test_legacy_kwargs_actually_enforce_boundaries(self):
        """Old kwargs must not just be accepted — they must work correctly."""
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            v = ToolCallBoundaryValidator(
                allowed_tools={"search"},
                forbidden_tools={"delete"},
            )
        run = _make_run(ToolCall(name="delete", arguments={}))
        result = v.evaluate_run(run)
        assert not result.passed
        assert any("delete" in f.description for f in result.findings)

    def test_boundary_and_legacy_kwargs_raises(self):
        import pytest

        with pytest.raises(ValueError, match="not both"):
            ToolCallBoundaryValidator(
                boundary=ToolBoundary(forbidden_tools={"x"}),
                forbidden_tools={"y"},
            )

    def test_no_legacy_kwargs_no_warning(self):
        """New API emits no warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ToolCallBoundaryValidator(boundary=ToolBoundary(forbidden_tools={"x"}))
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0
