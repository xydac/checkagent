"""Tests for checkagent.core.llm_call — shared LLM calling utility."""

import pytest

from checkagent.core.llm_call import detect_provider


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

    def test_unknown_model_raises_bad_parameter(self):
        import click

        with pytest.raises(click.BadParameter):
            detect_provider("gemini-pro")

    def test_unknown_model_raises_bad_parameter_2(self):
        import click

        with pytest.raises(click.BadParameter):
            detect_provider("llama-3-70b")

    def test_custom_param_hint(self):
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            detect_provider("unknown-model", param_hint="--my-flag")
        assert exc_info.value.param_hint == "--my-flag"
