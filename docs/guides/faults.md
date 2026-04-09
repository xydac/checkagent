# Fault Injection

Test how your agent handles failures. CheckAgent's fault injection system simulates timeouts, rate limits, malformed responses, and other failure modes.

## Quick Start

```python
@pytest.mark.agent_test(layer="mock")
async def test_handles_timeout(ap_mock_llm, ap_mock_tool, ap_fault):
    # Configure faults
    ap_fault.on_tool("external_api").timeout(seconds=30)

    # Wire faults into mocks
    ap_mock_tool.attach_faults(ap_fault)

    # Your agent should handle the timeout gracefully
    result = await my_agent.run("Fetch data", tools=ap_mock_tool)
    assert result.error is not None or result.final_output is not None
```

## Tool Faults

Inject faults on specific tools:

```python
# Timeout after N seconds
ap_fault.on_tool("slow_api").timeout(seconds=10)

# Rate limit after N calls
ap_fault.on_tool("rate_limited_api").rate_limit(after_n=3)

# Return malformed data
ap_fault.on_tool("flaky_api").returns_malformed({"corrupted": True})

# Return empty response
ap_fault.on_tool("empty_api").returns_empty()

# Intermittent failures (50% fail rate)
ap_fault.on_tool("unstable_api").intermittent(fail_rate=0.5, seed=42)

# Slow responses
ap_fault.on_tool("laggy_api").slow(latency_ms=2000)
```

## LLM Faults

Inject faults on LLM calls:

```python
# Context window overflow
ap_fault.on_llm().context_overflow()

# Partial/truncated response
ap_fault.on_llm().partial_response()

# Rate limit
ap_fault.on_llm().rate_limit(after_n=5)

# Server error
ap_fault.on_llm().server_error(message="Internal server error")

# Content filter triggered
ap_fault.on_llm().content_filter()

# Intermittent failures
ap_fault.on_llm().intermittent(fail_rate=0.3, seed=42)
```

## Attaching Faults to Mocks

Faults fire automatically when attached to `MockLLM` or `MockTool`:

```python
ap_mock_llm.attach_faults(ap_fault)
ap_mock_tool.attach_faults(ap_fault)

# Now every call checks for configured faults before returning
await ap_mock_llm.complete("hello")   # May raise if LLM fault triggers
await ap_mock_tool.call("search", {}) # May raise if tool fault triggers
```

## Inspecting Fault Triggers

After running your agent, check which faults fired:

```python
assert ap_fault.triggered  # At least one fault fired
assert ap_fault.trigger_count == 2
assert ap_fault.was_triggered("slow_api")

# Get detailed records
for record in ap_fault.triggered_records:
    print(f"{record.target}: {record.fault_type}")
```

## Chaining Faults

Configure multiple faults in a single chain:

```python
ap_fault.on_tool("api").timeout(seconds=5)
ap_fault.on_tool("api").intermittent(fail_rate=0.3)
ap_fault.on_llm().rate_limit(after_n=10)
```
