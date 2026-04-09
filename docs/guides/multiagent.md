# Multi-Agent Testing

Test systems where multiple agents collaborate, delegate, and hand off tasks.

## Capturing Multi-Agent Traces

```python
from checkagent import AgentRun, MultiAgentTrace

trace = MultiAgentTrace()

# Add runs for each agent
orchestrator_run = AgentRun(
    input="Plan a trip to Tokyo",
    run_id="run-1",
    agent_id="orchestrator",
    agent_name="Planner",
    final_output="Trip planned",
    steps=[...],
)
search_run = AgentRun(
    input="Find flights to Tokyo",
    run_id="run-2",
    agent_id="searcher",
    agent_name="FlightSearch",
    parent_run_id="run-1",
    final_output="Found 3 flights",
    steps=[...],
)

trace.add_run(orchestrator_run).add_run(search_run)
```

## Handoff Tracking

Record handoffs between agents:

```python
from checkagent.multiagent import Handoff, HandoffType

trace.add_handoff(Handoff(
    from_agent="orchestrator",
    to_agent="searcher",
    handoff_type=HandoffType.DELEGATION,
    payload={"task": "Find flights"},
))
```

Auto-detect handoffs from parent-child relationships:

```python
trace.detect_handoffs()  # Infers handoffs from parent_run_id
```

## Topology Queries

Inspect the agent graph:

```python
# Get agent hierarchy
children = trace.get_children("run-1")  # Agents spawned by orchestrator

# Trace the handoff chain
chain = trace.handoff_chain("orchestrator")  # ["orchestrator", "searcher"]

# Get all runs for an agent
runs = trace.get_runs_by_agent("searcher")
```

## Credit Assignment

When a multi-agent system fails, find which agent is responsible:

```python
from checkagent.multiagent import assign_blame, BlameStrategy

# Blame the leaf agents that errored (most useful for root cause analysis)
blame = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
for agent_id, reason in blame.items():
    print(f"{agent_id}: {reason}")
```

Blame strategies:

| Strategy | Description |
|----------|-------------|
| `LEAF_ERRORS` | Blame leaf agents (no outgoing handoffs) that errored |
| `FIRST_ERROR` | Blame the first agent in the chain that errored |
| `ALL_ERRORS` | Blame every agent that errored |
