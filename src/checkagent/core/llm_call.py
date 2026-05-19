"""Shared async LLM calling utility for checkagent CLI commands.

Provides a thin async wrapper over OpenAI and Anthropic clients so
CLI commands (scan, analyze-prompt) can call LLMs without duplicating
provider-detection and client-creation logic.
"""

from __future__ import annotations

import click


def detect_provider(model: str, *, param_hint: str = "--llm") -> str:
    """Return 'openai' or 'anthropic' from a model name string.

    Raises click.BadParameter if the provider cannot be determined.
    """
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    raise click.BadParameter(
        f"Cannot detect provider from model '{model}'. "
        "Use a model like 'gpt-4o-mini' (OpenAI) or 'claude-haiku-4-5-20251001' (Anthropic).",
        param_hint=param_hint,
    )


def check_api_key(model: str) -> str | None:
    """Return the API key environment variable name if it's missing, else None."""
    import os  # noqa: PLC0415

    provider = detect_provider(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return "OPENAI_API_KEY"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY"
    return None


async def call_llm(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 150,
    temperature: float = 0,
) -> str:
    """Call an LLM and return the raw text response.

    Detects provider from the model name.  Raises click.ClickException
    if the required SDK is not installed.
    """
    provider = detect_provider(model)

    if provider == "openai":
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise click.ClickException(
                "The 'openai' package is required for OpenAI models.\n"
                "Install it with:  pip install openai"
            ) from exc
        async with openai.AsyncOpenAI() as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return response.choices[0].message.content or ""

    # provider == "anthropic"
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise click.ClickException(
            "The 'anthropic' package is required for Claude models.\n"
            "Install it with:  pip install anthropic"
        ) from exc
    async with anthropic.AsyncAnthropic() as client:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    return response.content[0].text if response.content else ""
