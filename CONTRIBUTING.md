# Contributing to CheckAgent

Thanks for your interest in CheckAgent! We're building this in public and welcome contributions from day one.

## Getting Started

```bash
git clone https://github.com/xydac/checkagent.git
cd checkagent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## Development Process

1. Check existing issues or open a new one to discuss your idea
2. Fork the repo and create a branch from `main`
3. Write your code and tests
4. Run `pytest tests/` and make sure everything passes
5. Submit a PR with a clear description of what and why

## Code Conventions

- **Async-first** — all agent-facing APIs use `async def`
- **Fixtures** use `ca_` prefix (e.g., `ca_mock_llm`, `ca_fault`)
- **Adapters** stay under 200 lines — thin wrappers, not deep integrations
- **Types** go in `src/checkagent/core/types.py`
- **Tests** mirror the source layout in `tests/`

## Testing

- Layer 1 (mock) tests must run in < 100ms
- Layer 2 (replay) tests must run in < 1s
- Zero flaky tests in Layers 1 and 2
- Run `pytest tests/` before submitting a PR

## Commit Messages

Use clear, imperative commit messages:
- `Add mock LLM streaming support`
- `Fix cassette replay matching for tool calls`
- `Update safety probe templates for indirect injection`

## DCO Sign-Off

We use the Developer Certificate of Origin (DCO). Sign off your commits:

```bash
git commit -s -m "Your commit message"
```

This adds `Signed-off-by: Your Name <your@email.com>` to the commit.

## What to Contribute

- **Bug reports** — open an issue with reproduction steps
- **Bug fixes** — PRs welcome, include a test that catches the bug
- **New evaluators** — custom metrics via the plugin interface
- **New adapters** — framework integrations (prefer as separate packages for community adapters)
- **Safety probes** — new attack templates (must be non-destructive)
- **Documentation** — guides, examples, typo fixes
- **Examples** — real-world agent test suites

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
