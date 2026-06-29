# Replay Layer

The replay layer records real API interactions as JSON cassettes, then replays them for deterministic regression testing. After the initial recording, tests are free and fast.

## How It Works

1. **Record:** Run your agent against real APIs. CheckAgent captures every request and response as a JSON cassette file.
2. **Replay:** On subsequent test runs, the replay engine serves recorded responses instead of making real API calls.
3. **Detect regressions:** If your agent's behavior changes (different tool calls, different request patterns), the test fails — catching regressions before they ship.

## Recording a Cassette

Use the CLI to record a session:

```bash
checkagent record my_agent "Book a flight to Tokyo" --output tests/cassettes/booking.json
```

Or record programmatically:

```python
from checkagent.replay import CassetteRecorder, Cassette

recorder = CassetteRecorder()
recorder.record_interaction(request, response)
cassette = recorder.to_cassette()
cassette.save("tests/cassettes/booking.json")
```

## Replay in Tests

### Using the `ap_cassette` fixture (recommended)

The `ap_cassette` fixture handles recording and replaying automatically:

- **First run (record mode):** no cassette file exists → `ap_cassette.is_recording()` is `True`. Run your agent and record interactions via `ap_cassette.recorder`. After the test, the cassette is saved automatically.
- **Subsequent runs (replay mode):** cassette file found → `ap_cassette.is_replaying()` is `True`. Use `ap_cassette.engine` to match requests against recorded responses. No real API calls are made.

```python
from checkagent.replay import CassetteRecorder, RecordedRequest

@pytest.mark.agent_test(layer="replay")
async def test_booking_regression(ap_cassette, my_agent):
    if ap_cassette.is_recording():
        # Record mode: run your agent and capture interactions
        result = await my_agent.run("Book a flight to Tokyo")
        ap_cassette.recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": "Book a flight to Tokyo"}]},
            response_body={"choices": [{"message": {"content": "Booking confirmed!"}}]},
            model="gpt-4o",
        )
        assert "Booking" in result
    else:
        # Replay mode: verify agent uses recorded responses
        assert ap_cassette.cassette is not None
        assert len(ap_cassette.cassette.interactions) > 0
        # Run against replayed data — no live API calls
        result = await my_agent.run("Book a flight to Tokyo")
        assert "Booking" in result
```

The cassette is saved to `cassettes/<test_module>/<test_name>.json` by default. Override the path with a marker:

```python
@pytest.mark.cassette(path="tests/cassettes/booking_v2.json")
@pytest.mark.agent_test(layer="replay")
async def test_booking_v2(ap_cassette):
    ...
```

Add `tests/**/cassettes/` to `.gitignore` to keep recorded cassettes out of version control during development (or commit them to lock regression baselines).

### Manual replay

```python
from checkagent.replay import Cassette, ReplayEngine

@pytest.mark.agent_test(layer="replay")
async def test_booking_regression():
    cassette = Cassette.load("tests/cassettes/booking.json")
    engine = ReplayEngine(cassette)

    # Use engine.match() to serve recorded responses
    interaction = engine.match(request)
    assert interaction is not None
```

## Matching Strategies

The replay engine supports three matching strategies:

| Strategy | Description | Use When |
|----------|-------------|----------|
| `EXACT` | Method + body must match exactly | Strict regression tests |
| `SUBSET` | Recorded fields are a subset of request | Tolerant of extra fields |
| `SEQUENCE` | Match by position in the interaction sequence | Order-dependent flows |

```python
from checkagent.replay import MatchStrategy

engine = ReplayEngine(cassette, strategy=MatchStrategy.SUBSET)
```

For `SEQUENCE` mode, enable `strict_kind=True` to verify that LLM and tool calls arrive in the correct order. Useful when an agent mixes LLM calls and tool calls in a fixed sequence:

```python
# Raises CassetteMismatchError if the call kind (llm/tool) doesn't match
engine = ReplayEngine(cassette, strategy=MatchStrategy.SEQUENCE, strict_kind=True)
```

## Passthrough Mode

Allow unmatched requests to pass through to the real service:

```python
engine = ReplayEngine(cassette, block_unmatched=False)
result = engine.match(new_request)  # Returns None if no match
```

With `block_unmatched=True` (the default), unmatched requests raise `CassetteMismatchError`.

## Cassette Format

Cassettes are JSON files with a metadata block and a list of interactions:

```json
{
  "_meta": {
    "schema_version": 1,
    "recorded_at": "2025-01-01T00:00:00Z",
    "content_hash": "abc123..."
  },
  "interactions": [
    {
      "request": {"kind": "llm", "method": "complete", "body": {"text": "..."}},
      "response": {"body": {"text": "..."}, "status": "ok"}
    }
  ]
}
```

Cassettes use content-addressed filenames for git-friendliness and include automatic secret redaction.

## Migrating Cassettes

When the cassette schema evolves, use the migration tool:

```bash
checkagent migrate-cassettes tests/cassettes/
```
