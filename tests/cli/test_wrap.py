"""Tests for ``checkagent wrap`` CLI command."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from checkagent.cli.wrap import _detect_kind, _resolve_object, wrap_cmd

# ---------------------------------------------------------------------------
# Fixtures: in-memory agent objects for detection tests
# ---------------------------------------------------------------------------


class RunAgent:
    def run(self, prompt: str) -> str:
        return f"run: {prompt}"


class InvokeAgent:
    def invoke(self, prompt: str) -> str:
        return f"invoke: {prompt}"


class KickoffAgent:
    def kickoff(self, inputs: dict) -> str:
        return f"kickoff: {inputs}"


class MultiMethodAgent:
    """Has both .run() and .invoke() — .run() takes priority."""

    def run(self, prompt: str) -> str:
        return f"run: {prompt}"

    def invoke(self, prompt: str) -> str:
        return f"invoke: {prompt}"


async def plain_callable(prompt: str) -> str:
    return f"plain: {prompt}"


class NotAnAgent:
    """Neither callable nor has any known method."""

    x = 42


# ---------------------------------------------------------------------------
# _detect_kind unit tests
# ---------------------------------------------------------------------------


class TestDetectKind:
    def test_run_method(self):
        assert _detect_kind(RunAgent()) == "run"

    def test_invoke_method(self):
        assert _detect_kind(InvokeAgent()) == "invoke"

    def test_kickoff_method(self):
        assert _detect_kind(KickoffAgent()) == "kickoff"

    def test_run_takes_priority_over_invoke(self):
        assert _detect_kind(MultiMethodAgent()) == "run"

    def test_plain_callable(self):
        assert _detect_kind(plain_callable) == "callable"

    def test_lambda_callable(self):
        assert _detect_kind(lambda p: p) == "callable"

    def test_unknown_raises(self):
        import click

        with pytest.raises(click.ClickException, match="Cannot determine"):
            _detect_kind(NotAnAgent())


# ---------------------------------------------------------------------------
# Helper: write a temp module for _resolve_object / wrap_cmd tests
# ---------------------------------------------------------------------------


def _write_agent_module(tmp_path: Path) -> Path:
    mod = tmp_path / "wrap_agents.py"
    mod.write_text(
        textwrap.dedent("""\
            class RunAgent:
                def run(self, prompt):
                    return f"run:{prompt}"

            class InvokeAgent:
                def invoke(self, prompt):
                    return f"invoke:{prompt}"

            class KickoffAgent:
                def kickoff(self, inputs):
                    return f"kickoff:{inputs}"

            async def plain_fn(prompt):
                return f"plain:{prompt}"

            run_instance = RunAgent()
            invoke_instance = InvokeAgent()
            kickoff_instance = KickoffAgent()

            not_callable = 42
        """),
        encoding="utf-8",
    )
    return mod


# ---------------------------------------------------------------------------
# _resolve_object tests
# ---------------------------------------------------------------------------


class TestResolveObject:
    def test_colon_syntax(self, tmp_path: Path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        obj, mod, attr = _resolve_object("wrap_agents:RunAgent")
        assert attr == "RunAgent"
        assert mod == "wrap_agents"

    def test_dot_syntax(self, tmp_path: Path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        obj, mod, attr = _resolve_object("wrap_agents.RunAgent")
        assert attr == "RunAgent"

    def test_missing_module_raises(self, tmp_path: Path, monkeypatch):
        import click

        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.ClickException, match="Cannot import"):
            _resolve_object("no_such_module:Foo")

    def test_missing_attr_raises(self, tmp_path: Path, monkeypatch):
        import click

        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.ClickException, match="has no attribute"):
            _resolve_object("wrap_agents:DoesNotExist")

    def test_invalid_syntax_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            _resolve_object("nodotsorcolons")


# ---------------------------------------------------------------------------
# wrap_cmd integration tests (via CliRunner + tmp module)
# ---------------------------------------------------------------------------


class TestWrapCommand:
    def _run(self, tmp_path: Path, monkeypatch, args: list[str]):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        return runner.invoke(wrap_cmd, args, catch_exceptions=False)

    # --- .run() detection ---

    def test_run_generates_file(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:run_instance"])
        assert result.exit_code == 0
        out = tmp_path / "checkagent_target.py"
        assert out.exists()
        assert ".run(" in out.read_text()

    def test_run_output_contains_method_label(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:run_instance"])
        assert ".run()" in result.output

    def test_run_class_generates_file(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:RunAgent"])
        assert result.exit_code == 0
        assert ".run(" in (tmp_path / "checkagent_target.py").read_text()

    # --- .invoke() detection ---

    def test_invoke_generates_file(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:invoke_instance"])
        assert result.exit_code == 0
        assert ".invoke(" in (tmp_path / "checkagent_target.py").read_text()

    def test_invoke_output_label(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:invoke_instance"])
        assert ".invoke()" in result.output

    # --- .kickoff() detection ---

    def test_kickoff_generates_file(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:kickoff_instance"])
        assert result.exit_code == 0
        content = (tmp_path / "checkagent_target.py").read_text()
        assert ".kickoff(" in content
        assert "inputs" in content

    def test_kickoff_uses_inputs_dict(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, ["wrap_agents:kickoff_instance"])
        content = (tmp_path / "checkagent_target.py").read_text()
        assert 'inputs={"prompt": prompt}' in content

    # --- plain callable ---

    def test_callable_prints_no_wrapper_needed(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:plain_fn"])
        assert result.exit_code == 0
        assert "No wrapper needed" in result.output
        assert not (tmp_path / "checkagent_target.py").exists()

    def test_callable_suggests_scan_directly(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ["wrap_agents:plain_fn"])
        assert "checkagent scan" in result.output

    # --- --output flag ---

    def test_custom_output_path(self, tmp_path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            wrap_cmd,
            ["wrap_agents:run_instance", "--output", "my_wrapper.py"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (tmp_path / "my_wrapper.py").exists()

    # --- --force flag ---

    def test_refuses_to_overwrite_without_force(self, tmp_path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkagent_target.py").write_text("# existing")
        runner = CliRunner()
        result = runner.invoke(wrap_cmd, ["wrap_agents:run_instance"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_force_overwrites_existing(self, tmp_path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkagent_target.py").write_text("# existing")
        runner = CliRunner()
        result = runner.invoke(
            wrap_cmd, ["wrap_agents:run_instance", "--force"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "# existing" not in (tmp_path / "checkagent_target.py").read_text()

    # --- error cases ---

    def test_bad_target_syntax(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(wrap_cmd, ["nodotsorcolons"])
        assert result.exit_code != 0

    def test_missing_module(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(wrap_cmd, ["no_such_module:Foo"])
        assert result.exit_code != 0

    # --- generated file is valid Python ---

    def test_generated_run_wrapper_is_valid_python(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, ["wrap_agents:run_instance"])
        code = (tmp_path / "checkagent_target.py").read_text()
        compile(code, "checkagent_target.py", "exec")

    def test_generated_invoke_wrapper_is_valid_python(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, ["wrap_agents:invoke_instance"])
        code = (tmp_path / "checkagent_target.py").read_text()
        compile(code, "checkagent_target.py", "exec")

    def test_generated_kickoff_wrapper_is_valid_python(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, ["wrap_agents:kickoff_instance"])
        code = (tmp_path / "checkagent_target.py").read_text()
        compile(code, "checkagent_target.py", "exec")

    # --- registered in main CLI group ---

    def test_wrap_registered_in_main_cli(self):
        from checkagent.cli import main

        assert "wrap" in main.commands

    def test_wrap_shows_in_help(self):
        from checkagent.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "wrap" in result.output


# ---------------------------------------------------------------------------
# agents.Agent detection (mocked — package may not be installed)
# ---------------------------------------------------------------------------


class TestAgentsSDKDetection:
    def test_agents_agent_detected_as_agents_runner(self, monkeypatch):
        """If 'agents' package is present and obj is agents.Agent, use Runner.run()."""

        class FakeAgent:
            pass

        class FakeAgentsModule:
            Agent = FakeAgent

        monkeypatch.setitem(
            __import__("sys").modules, "agents", FakeAgentsModule()
        )

        instance = FakeAgent()
        assert _detect_kind(instance) == "agents_runner"

    def test_agents_runner_template_uses_runner_run(self, tmp_path, monkeypatch):
        """Generated wrapper must use Runner.run()."""

        class FakeAgent:
            pass

        class FakeAgentsModule:
            Agent = FakeAgent

        import sys as _sys

        monkeypatch.setitem(_sys.modules, "agents", FakeAgentsModule())

        # Write a module that has an agents.Agent instance
        mod = tmp_path / "fake_agent_mod.py"
        mod.write_text("agent = None\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        import importlib

        original_import = importlib.import_module

        def patched_import(name, *args, **kwargs):
            if name == "fake_agent_mod":
                import types

                m = types.ModuleType("fake_agent_mod")
                m.agent = FakeAgent()
                return m
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", patched_import)

        runner = CliRunner()
        result = runner.invoke(
            wrap_cmd, ["fake_agent_mod:agent"], catch_exceptions=False
        )
        assert result.exit_code == 0
        content = (tmp_path / "checkagent_target.py").read_text()
        assert "Runner.run(" in content
        assert "Runner.run()" in result.output
