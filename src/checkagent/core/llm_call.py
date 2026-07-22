"""Shared async LLM calling utility for checkagent CLI commands.

Provides a thin async wrapper over OpenAI and Anthropic clients so
CLI commands (scan, analyze-prompt) can call LLMs without duplicating
provider-detection and client-creation logic.

Supported providers:
  - openai   — gpt-4o-mini, gpt-4o, o1, o3, o4 prefixes
  - anthropic — claude- prefix (requires ANTHROPIC_API_KEY)
  - claude-code — uses local Claude Code CLI; no API key required
"""

from __future__ import annotations

import asyncio
import subprocess

import click


def detect_provider(model: str, *, param_hint: str = "--llm") -> str:
    """Return 'openai', 'anthropic', or 'claude-code' from a model name string.

    Raises click.BadParameter if the provider cannot be determined.
    """
    if model == "claude-code":
        return "claude-code"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    raise click.BadParameter(
        f"Cannot detect provider from model '{model}'. "
        "Use a model like 'gpt-4o-mini' (OpenAI), 'claude-haiku-4-5-20251001' (Anthropic), "
        "or 'claude-code' (local Claude Code CLI, no API key required).",
        param_hint=param_hint,
    )


def check_api_key(model: str) -> str | None:
    """Return the API key env var name if it's missing, else None.

    Returns None for 'claude-code' (uses local CLI, no key needed).
    """
    import os  # noqa: PLC0415

    provider = detect_provider(model)
    if provider == "claude-code":
        return None
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return "OPENAI_API_KEY"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY"
    return None


def _invoke_claude_cli(system: str, user: str) -> str:
    """Shell out to the local Claude Code CLI and return the response text.

    Uses ``claude --print --bare --system-prompt <system> <user>``.
    Raises RuntimeError if the CLI is not found or exits non-zero.
    """
    import shutil  # noqa: PLC0415

    if not shutil.which("claude"):
        raise click.ClickException(
            "The 'claude' CLI is required for --llm claude-code.\n"
            "Install Claude Code: https://claude.ai/code"
        )
    result = subprocess.run(
        ["claude", "--print", "--bare", "--system-prompt", system, user],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:300] or "claude CLI exited non-zero")
    return result.stdout.strip()


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
    if the required SDK is not installed or the CLI is missing.
    """
    provider = detect_provider(model)

    if provider == "claude-code":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _invoke_claude_cli, system, user)

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
