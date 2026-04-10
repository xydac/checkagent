# Mock Layer

The mock layer lets you test agent logic without making any LLM or tool API calls. Tests are free, deterministic, and run in milliseconds.

## MockLLM

`ca_mock_llm` provides a mock LLM that returns deterministic responses based on pattern matching.

### Basic Usage

```python
@pytest.mark.agent_test(layer="mock")
async def test_agent(ca_mock_llm):
    # Set up a response rule
    ca_mock_llm.on_input(contains="weather").respond("It's sunny today.")

    # Your agent calls the mock LLM
    response = await ca_mock_llm.complete("What's the weather?")
    assert response == "It's sunny today."
```

### Pattern Matching

Match inputs by substring, regex, or exact match:

```python
# Substring match (default)
ca_mock_llm.on_input(contains="book").respond("Booking confirmed.")

# Regex match
ca_mock_llm.on_input(pattern=r"flight to \w+").respond("Searching flights...")

# Exact match
ca_mock_llm.on_input(exact="hello").respond("Hi there!")
```

Rules are checked in order — the first match wins.

### Multiple Responses

Pass a list to cycle through responses on successive calls:

```python
ca_mock_llm.on_input(contains="step").respond([
    "First, I'll search for information.",
    "Now I'll summarize what I found.",
    "Here's my final answer.",
])
```

### Streaming

Mock streaming responses that yield chunks:

```python
ca_mock_llm.on_input(contains="story").stream(
    ["Once ", "upon ", "a ", "time..."],
    delay_ms=10,
)

chunks = []
async for event in ca_mock_llm.stream("Tell me a story"):
    if event.data:
        chunks.append(event.data)
assert "".join(chunks) == "Once upon a time..."
```

### Token Usage Tracking

Track token usage across calls:

```python
ca_mock_llm.with_usage(auto_estimate=True)

await ca_mock_llm.complete("Hello world")
assert ca_mock_llm.last_call.prompt_tokens > 0
assert ca_mock_llm.last_call.completion_tokens > 0
```

### Inspecting Calls

```python
await ca_mock_llm.complete("first call")
await ca_mock_llm.complete("second call")

assert ca_mock_llm.call_count == 2
assert ca_mock_llm.was_called_with("first")
assert ca_mock_llm.last_call.input_text == "second call"

# Get all calls matching a pattern
matching = ca_mock_llm.get_calls_matching("call")
assert len(matching) == 2
```

## MockTool

`ca_mock_tool` provides a mock tool executor with schema validation and call recording.

### Basic Usage

```python
@pytest.mark.agent_test(layer="mock")
async def test_tool_use(ca_mock_tool):
    ca_mock_tool.on_call("search").respond({"results": ["result1", "result2"]})

    result = await ca_mock_tool.call("search", {"query": "test"})
    assert result == {"results": ["result1", "result2"]}
```

### Schema Validation

Define a JSON schema to validate tool arguments:

```python
ca_mock_tool.on_call("create_event").respond(
    {"id": "evt-1"},
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["title"],
    },
)

# This passes validation
await ca_mock_tool.call("create_event", {"title": "Meeting", "date": "2025-01-01"})

# This raises ValidationError — missing required "title"
await ca_mock_tool.call("create_event", {"date": "2025-01-01"})
```

### Error Responses

Simulate tool errors:

```python
ca_mock_tool.on_call("flaky_api").error("Service temporarily unavailable")

try:
    await ca_mock_tool.call("flaky_api", {})
except Exception:
    pass  # Your agent should handle this
```

### Assertions

Assert that tools were called correctly:

```python
await my_agent.run("Book a flight", tools=ca_mock_tool)

# Assert tool was called
ca_mock_tool.assert_tool_called("search_flights")

# Assert with specific arguments
ca_mock_tool.assert_tool_called("search_flights", with_args={"destination": "Tokyo"})

# Assert call count
ca_mock_tool.assert_tool_called("search_flights", times=1)

# Assert tool was NOT called
ca_mock_tool.assert_tool_not_called("delete_account")
```

You can also use the top-level `assert_tool_called` on an `AgentRun`:

```python
from checkagent import assert_tool_called

result = await my_agent.run("Book a flight", tools=ca_mock_tool)
assert_tool_called(result, "search_flights", destination="Tokyo")
```

## Structured Output Assertions

Assert on agent output structure using Pydantic models or dictionaries:

```python
from checkagent import assert_output_schema, assert_output_matches
from pydantic import BaseModel

class BookingResult(BaseModel):
    confirmed: bool
    booking_id: str

# Validate against a Pydantic model
assert_output_schema(result, BookingResult)

# Validate against a partial dictionary (unmentioned keys are ignored)
assert_output_matches(result, {"confirmed": True})
```

`assert_output_matches` supports [dirty-equals](https://dirty-equals.helpmanual.io/) for flexible matching:

```python
from dirty_equals import IsStr, IsPositiveInt

assert_output_matches(result, {
    "booking_id": IsStr(regex=r"^BK-\d+$"),
    "seats": IsPositiveInt,
})
```

## Default Behavior

If no rule matches an input, `MockLLM` returns its `default_response` (default: `"Mock response"`). You can change this:

```python
llm = MockLLM(default_response="I don't know.")
```

Similarly, `MockTool` returns its `default_response` (default: `None`) for unregistered tools when `strict_validation=False`. With strict validation (the default), calling an unregistered tool raises an error.
