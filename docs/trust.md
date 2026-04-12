# Trust & Privacy

Developers ask this before adopting any dev tool: *what does it phone home?*

Short answer: nothing. Here is the full picture.

---

## Zero telemetry

!!! note "Zero telemetry"
    CheckAgent never collects usage data, crash reports, analytics, or any other telemetry. There is no tracking of any kind — no anonymous event pings, no opt-in/opt-out toggle, no background process.

There is no telemetry infrastructure to send data to. The package contains no HTTP calls except the ones you explicitly trigger by running an eval or judge layer against an LLM API of your choosing.

---

## Self-hosted by design

CheckAgent is a pytest plugin. It runs as a Python process on whatever machine runs your tests: your laptop, your CI runner, an air-gapped build server. There is no CheckAgent cloud service, no remote scheduler, no intermediary.

- No account required
- No login
- No SaaS dependency
- Works fully air-gapped (mock and replay layers need no network access at all)

The GitHub Action (`xydac/checkagent`) is a [composite action](https://docs.github.com/en/actions/creating-actions/creating-a-composite-action) — it runs inside your own GitHub Actions runner, not on infrastructure operated by CheckAgent.

---

## Your API keys stay yours

When you configure an LLM API key for the eval or judge layers, that key is passed directly from your environment to the LLM provider (OpenAI, Anthropic, etc.). CheckAgent:

- Does not proxy API calls through any intermediary
- Does not store keys in any file or database
- Does not log keys or include them in any output

The mock (Layer 1) and replay (Layer 2) layers require no API keys at all. Keys only enter the picture when you explicitly opt into eval or judge layers.

---

## Your test data stays local

Cassettes, golden datasets, and SARIF reports are ordinary files written to your project directory. CheckAgent has no server to send them to.

| Artifact | Location | Who can see it |
|---|---|---|
| Cassettes | `cassettes/` in your repo | You and your team |
| Golden datasets | wherever you configure | You and your team |
| SARIF reports | your repo or CI artifacts | You and your team |

SARIF files uploaded to GitHub Code Scanning go to **your repository's** Security tab via the standard GitHub API — not to any CheckAgent endpoint.

---

## Open source

!!! note "Apache-2.0"
    CheckAgent is [Apache-2.0](https://github.com/xydac/checkagent/blob/main/LICENSE). There are no closed binary components, no obfuscated code, no "community edition" restrictions.

Read the source at [github.com/xydac/checkagent](https://github.com/xydac/checkagent). The entire framework is in `src/checkagent/`. If you want to verify a specific claim on this page, the code is the authoritative answer.

---

## SARIF: your data, your format

[SARIF](https://sarifweb.azurewebsites.net/) (Static Analysis Results Interchange Format) is an open standard (OASIS SARIF TC, referenced by ISO/IEC). CheckAgent emits standard SARIF 2.1.0 — a plain JSON file with a well-defined schema.

What you do with the file is entirely up to you:

- Upload to GitHub Code Scanning
- Store in S3 or your artifact registry
- Parse it with any SARIF-aware tool
- Share with your security team
- Archive it alongside your build artifacts

CheckAgent does not read, index, or receive these files after they are written.

---

## Verifying it yourself

If you want to confirm these claims rather than take them on faith:

**Read the source.** Network calls live in `src/checkagent/adapters/` (adapter code that calls LLM APIs when you ask it to). There is no other outbound HTTP in the codebase. Verify with:

```bash
grep -r "requests\|httpx\|urllib\|aiohttp" src/checkagent/ --include="*.py"
```

**Run with network monitoring.** Tools like `mitmproxy`, Wireshark, or `nettop` (macOS) let you observe outbound connections during a test run. A mock-layer run should produce zero outbound network traffic.

**Inspect the dependency tree.** `pip show checkagent` and `pip install checkagent --dry-run` show exactly what gets installed. Cross-reference against the dependency list below.

---

## Third-party dependencies

CheckAgent's required dependencies are all permissive, widely-audited packages. None of them phone home.

| Package | License | Purpose |
|---|---|---|
| [pytest](https://github.com/pytest-dev/pytest) | MIT | Test runner |
| [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) | Apache-2.0 | Async test support |
| [pluggy](https://github.com/pytest-dev/pluggy) | MIT | Plugin system (shared with pytest) |
| [pydantic](https://github.com/pydantic/pydantic) | MIT | Config and type validation |
| [click](https://github.com/pallets/click) | BSD-3-Clause | CLI |
| [rich](https://github.com/Textualize/rich) | MIT | Terminal output formatting |

Optional dependencies (`opentelemetry-api`, `dirty-equals`, `deepdiff`, `spacy`) are only installed if you explicitly request them and are governed by their own permissive licenses.

---

## Questions?

Open an issue at [github.com/xydac/checkagent/issues](https://github.com/xydac/checkagent/issues). Questions about trust and privacy are treated as first-class issues, not noise.
