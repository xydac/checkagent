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
