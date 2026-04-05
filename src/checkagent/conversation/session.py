"""Multi-turn conversation session for testing agent dialogues.

Manages stateful conversation sessions where each turn accumulates
context. The agent under test sees the full conversation history.

Implements F10.1, F10.3, F10.4 from the PRD.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from checkagent.core.types import AgentInput, AgentRun, ToolCall


class Turn:
    """A single turn in a conversation with its input and result."""

    def __init__(self, index: int, input_text: str, result: AgentRun) -> None:
        self.index = index
        self.input_text = input_text
        self.result = result

    @property
    def output_text(self) -> str | None:
        """The final output text of this turn."""
        output = self.result.final_output
        if output is None:
            return None
        return str(output)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls made during this turn."""
        return self.result.tool_calls


class Conversation:
    """A stateful multi-turn conversation session.

    Wraps an async callable (agent function) and manages conversation
    history across multiple turns. Each call to ``say()`` sends the
    full conversation history to the agent.

    Usage::

        conv = Conversation(agent_fn)
        r1 = await conv.say("Hello")
        r2 = await conv.say("What did I just say?")
        assert conv.total_turns == 2

    The ``agent_fn`` receives an ``AgentInput`` with
    ``conversation_history`` populated from prior turns.
    """

    def __init__(
        self,
        agent_fn: Callable[[AgentInput], Awaitable[AgentRun]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._agent_fn = agent_fn
        self._turns: list[Turn] = []
        self._metadata: dict[str, Any] = metadata or {}

    async def say(
        self,
        text: str,
        *,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Send a message and get the agent's response.

        Builds an ``AgentInput`` with the full conversation history
        from prior turns, calls the agent function, and records the
        result as a new turn.

        Returns the ``AgentRun`` trace for this turn.
        """
        history = self._build_history()

        agent_input = AgentInput(
            query=text,
            context=context or {},
            conversation_history=history,
            metadata=metadata or {},
        )

        result = await self._agent_fn(agent_input)

        turn = Turn(
            index=len(self._turns),
            input_text=text,
            result=result,
        )
        self._turns.append(turn)

        return result

    def _build_history(self) -> list[dict[str, str]]:
        """Build conversation history from prior turns."""
        history: list[dict[str, str]] = []
        for turn in self._turns:
            history.append({"role": "user", "content": turn.input_text})
            output = turn.output_text
            if output is not None:
                history.append({"role": "assistant", "content": output})
        return history

    # --- Properties ---

    @property
    def turns(self) -> list[Turn]:
        """All completed turns."""
        return list(self._turns)

    @property
    def total_turns(self) -> int:
        """Number of completed turns."""
        return len(self._turns)

    @property
    def total_tool_calls(self) -> int:
        """Total tool calls across all turns."""
        return sum(len(t.tool_calls) for t in self._turns)

    @property
    def total_steps(self) -> int:
        """Total steps across all turns."""
        return sum(len(t.result.steps) for t in self._turns)

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """Flat list of every tool call across all turns."""
        return [tc for t in self._turns for tc in t.tool_calls]

    @property
    def total_prompt_tokens(self) -> int | None:
        """Sum of prompt tokens across all turns, or None if any turn lacks data."""
        tokens = [t.result.total_prompt_tokens for t in self._turns]
        if not tokens or any(t is None for t in tokens):
            return None
        return sum(t for t in tokens if t is not None)

    @property
    def total_completion_tokens(self) -> int | None:
        """Sum of completion tokens across all turns, or None if any turn lacks data."""
        tokens = [t.result.total_completion_tokens for t in self._turns]
        if not tokens or any(t is None for t in tokens):
            return None
        return sum(t for t in tokens if t is not None)

    @property
    def total_tokens(self) -> int | None:
        """Total tokens across all turns."""
        p = self.total_prompt_tokens
        c = self.total_completion_tokens
        if p is not None and c is not None:
            return p + c
        return None

    @property
    def last_turn(self) -> Turn | None:
        """The most recent turn, or None."""
        return self._turns[-1] if self._turns else None

    @property
    def last_result(self) -> AgentRun | None:
        """The AgentRun from the most recent turn, or None."""
        return self._turns[-1].result if self._turns else None

    def get_turn(self, index: int) -> Turn:
        """Get a specific turn by index.

        Raises IndexError if the index is out of range.
        """
        return self._turns[index]

    def tool_was_called(self, name: str) -> bool:
        """Check if a tool was called in any turn."""
        return any(tc.name == name for tc in self.all_tool_calls)

    def tool_was_called_in_turn(self, turn_index: int, name: str) -> bool:
        """Check if a tool was called in a specific turn."""
        turn = self._turns[turn_index]
        return any(tc.name == name for tc in turn.tool_calls)

    def context_references(self, turn: int, references_turn: int) -> bool:
        """Check if a turn's output references content from an earlier turn.

        A simple heuristic: checks if the earlier turn's input text
        appears (as a substring) in the later turn's output text.
        This covers the common case where the agent echoes or references
        earlier user messages.

        For more sophisticated reference detection, use the judge layer.
        """
        if turn < 0 or turn >= len(self._turns):
            raise IndexError(f"Turn {turn} out of range (0-{len(self._turns) - 1})")
        if references_turn < 0 or references_turn >= len(self._turns):
            raise IndexError(
                f"Turn {references_turn} out of range (0-{len(self._turns) - 1})"
            )
        if references_turn >= turn:
            raise ValueError(
                f"references_turn ({references_turn}) must be before turn ({turn})"
            )

        later_output = self._turns[turn].output_text
        if later_output is None:
            return False

        earlier_input = self._turns[references_turn].input_text
        earlier_output = self._turns[references_turn].output_text

        # Check if the later turn references either the input or output
        # of the earlier turn
        later_lower = later_output.lower()
        if earlier_input.lower() in later_lower:
            return True
        return earlier_output is not None and earlier_output.lower() in later_lower

    def reset(self) -> None:
        """Clear all turns and start fresh."""
        self._turns.clear()
