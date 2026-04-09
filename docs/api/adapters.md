# Adapters

The adapter protocol and framework-specific implementations.

## AgentAdapter Protocol

::: checkagent.core.adapter.AgentAdapter

## GenericAdapter

::: checkagent.adapters.generic.GenericAdapter

## wrap

::: checkagent.adapters.generic.wrap

## Framework Adapters

Framework adapters require their respective packages to be installed.
Install with optional extras:

```bash
pip install checkagent[langchain]     # LangChain
pip install checkagent[openai-agents] # OpenAI Agents SDK
pip install checkagent[pydantic-ai]   # PydanticAI
pip install checkagent[crewai]        # CrewAI
pip install checkagent[anthropic]     # Anthropic
```

### LangChainAdapter

::: checkagent.adapters.langchain.LangChainAdapter

### OpenAIAgentsAdapter

::: checkagent.adapters.openai_agents.OpenAIAgentsAdapter

### PydanticAIAdapter

::: checkagent.adapters.pydantic_ai.PydanticAIAdapter

### CrewAIAdapter

::: checkagent.adapters.crewai.CrewAIAdapter

### AnthropicAdapter

::: checkagent.adapters.anthropic.AnthropicAdapter
