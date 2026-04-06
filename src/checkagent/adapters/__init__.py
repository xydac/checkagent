"""Adapters — framework-specific wrappers conforming to AgentAdapter."""

from checkagent.adapters.generic import GenericAdapter, wrap

__all__ = ["GenericAdapter", "LangChainAdapter", "OpenAIAgentsAdapter", "wrap"]


def __getattr__(name: str):
    """Lazy-load framework adapters to avoid import errors for optional deps."""
    if name == "LangChainAdapter":
        from checkagent.adapters.langchain import LangChainAdapter

        return LangChainAdapter
    if name == "OpenAIAgentsAdapter":
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        return OpenAIAgentsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
