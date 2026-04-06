"""Adapters — framework-specific wrappers conforming to AgentAdapter."""

from checkagent.adapters.generic import GenericAdapter, wrap

__all__ = [
    "AnthropicAdapter",
    "CrewAIAdapter",
    "GenericAdapter",
    "LangChainAdapter",
    "OpenAIAgentsAdapter",
    "PydanticAIAdapter",
    "wrap",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LangChainAdapter": ("checkagent.adapters.langchain", "LangChainAdapter"),
    "OpenAIAgentsAdapter": ("checkagent.adapters.openai_agents", "OpenAIAgentsAdapter"),
    "CrewAIAdapter": ("checkagent.adapters.crewai", "CrewAIAdapter"),
    "PydanticAIAdapter": ("checkagent.adapters.pydantic_ai", "PydanticAIAdapter"),
    "AnthropicAdapter": ("checkagent.adapters.anthropic", "AnthropicAdapter"),
}


def __getattr__(name: str):
    """Lazy-load framework adapters to avoid import errors for optional deps."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
