"""Tests for checkagent.core.llm_call — shared LLM calling utility."""

import asyncio
from unittest.mock import MagicMock, patch

import click
import pytest

from checkagent.core.llm_call import _invoke_claude_cli, check_api_key, detect_provider


class TestDetectProvider:
    def test_gpt_models_detected_as_openai(self):
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"

    def test_o1_models_detected_as_openai(self):
        assert detect_provider("o1-preview") == "openai"
        assert detect_provider("o3-mini") == "openai"
        assert detect_provider("o4-mini") == "openai"

    def test_claude_models_detected_as_anthropic(self):
        assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"
        assert detect_provider("claude-sonnet-4-6") == "anthropic"
        assert detect_provider("claude-opus-4-7") == "anthropic"

    def test_claude_code_detected_as_claude_code(self):
        assert detect_provider("claude-code") == "claude-code"

    def test_unknown_model_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            detect_provider("gemini-pro")

    def test_unknown_model_raises_bad_parameter_2(self):
        with pytest.raises(click.BadParameter):
            detect_provider("llama-3-70b")

    def test_custom_param_hint(self):
        with pytest.raises(click.BadParameter) as exc_info:
            detect_provider("unknown-model", param_hint="--my-flag")
        assert exc_info.value.param_hint == "--my-flag"

    def test_claude_code_in_error_message_for_unknown_model(self):
        with pytest.raises(click.BadParameter) as exc_info:
            detect_provider("foobar")
        assert "claude-code" in str(exc_info.value)


class TestCheckApiKey:
    def test_claude_code_needs_no_api_key(self):
        assert check_api_key("claude-code") is None

    def test_openai_missing_key_returned(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert check_api_key("gpt-4o-mini") == "OPENAI_API_KEY"

    def test_openai_present_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert check_api_key("gpt-4o-mini") is None

    def test_anthropic_missing_key_returned(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert check_api_key("claude-sonnet-4-6") == "ANTHROPIC_API_KEY"


class TestCallLlmClaudeCode:
    def test_call_llm_claude_code_dispatches_to_subprocess(self):
        from checkagent.core.llm_call import call_llm

        with patch(
            "checkagent.core.llm_call._invoke_claude_cli", return_value="mocked"
        ) as mock_fn:
            result = asyncio.run(call_llm("claude-code", "system", "user"))
        mock_fn.assert_called_once_with("system", "user")
        assert result == "mocked"

    def test_invoke_claude_cli_raises_when_not_installed(self):
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(click.ClickException, match="claude.*CLI"),
        ):
            _invoke_claude_cli("sys", "user")

    def test_invoke_claude_cli_raises_on_nonzero_exit(self):
        from checkagent.core.llm_call import _invoke_claude_cli as _fn

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "auth error"
        mock_result.stdout = ""

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="auth error"),
        ):
            _fn("sys", "user")

    def test_invoke_claude_cli_returns_stripped_stdout(self):
        from checkagent.core.llm_call import _invoke_claude_cli as _fn

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '  {"answer": "yes"}  \n'
        mock_result.stderr = ""

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("subprocess.run", return_value=mock_result),
        ):
            out = _fn("sys", "user")
        assert out == '{"answer": "yes"}'
