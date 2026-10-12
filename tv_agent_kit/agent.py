"""The companion agent.

`create_companion_agent()` builds a Strands (AWS) agent on Amazon Bedrock
with a companion system prompt and your tools. `CompanionAgent.ask()` is
the single entry point the server (and your TV app) uses, and it *always*
returns a validated `CompanionReply`.

Design: the model decides, the kit enforces. The system prompt carries the
policy (spoilers, tone, output shape); `parse_reply()` guarantees the wire
contract no matter what the model actually emits.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from .llm import (
    DEFAULT_FAST_MODEL,
    DEFAULT_REASONING_MODEL,
    BedrockLLMClient,
    LLMClient,
)
from .protocol import CompanionAction, CompanionReply, parse_reply

try:  # strands is an optional dependency at import time
    from strands import Agent as _StrandsAgent
    from strands.tools import tool as _strands_tool

    _HAS_STRANDS = True
except ImportError:  # pragma: no cover
    _StrandsAgent = None  # type: ignore
    _strands_tool = None  # type: ignore
    _HAS_STRANDS = False


def tool(fn: Callable) -> Callable:
    """Decorator for companion tools.

    With `strands-agents` installed this is the real `@tool` (docstring
    becomes the model-facing tool description). Without it, the function
    passes through unchanged so unit tests can run anywhere.
    """
    if _HAS_STRANDS:
        return _strands_tool(fn)
    return fn


class CompanionAgent:
    """A Strands agent wrapped in the companion contract."""

    def __init__(
        self,
        name: str = "companion",
        system_prompt: str = "",
        tools: Optional[List[Callable]] = None,
        model: str = DEFAULT_REASONING_MODEL,
        fast_model: str = DEFAULT_FAST_MODEL,
        llm: Optional[LLMClient] = None,
        strands_agent: Optional[Any] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.fast_model = fast_model
        self.llm: LLMClient = llm or BedrockLLMClient()
        self._tools = tools or []

        if strands_agent is not None:
            self._agent = strands_agent
        elif _HAS_STRANDS:
            self._agent = _StrandsAgent(
                name=name,
                model=model,
                system_prompt=system_prompt,
                tools=self._tools,
            )
        else:
            self._agent = None

    # ------------------------------------------------------------------

    def context_block(self, viewer_id: str, state: dict) -> str:
        """Viewer state injected into every turn (spoiler math lives here)."""
        if not state:
            return f"Viewer: {viewer_id}. No stored state yet."
        return (
            f"Viewer: {viewer_id}.\nStored viewer state (JSON):\n{json.dumps(state, ensure_ascii=False)}"
        )

    def ask(
        self,
        viewer_id: str,
        message: str,
        state: Optional[dict] = None,
        **kwargs: Any,
    ) -> CompanionReply:
        """Run one companion turn. Always returns a CompanionReply."""
        if self._agent is None:
            if _HAS_STRANDS is False:
                return CompanionReply.error(
                    "no_agent",
                    "The agent backend is not available. Install 'tv-agent-quickstart[bedrock]' "
                    "(strands-agents + boto3) and configure AWS credentials.",
                )
        prompt = (
            f"{self.context_block(viewer_id, state or {})}\n\n"
            f"User message: {message}"
        )
        try:
            raw = self._agent(prompt, **kwargs)
        except Exception as e:  # noqa: BLE001 — the TV must never crash
            return CompanionReply.error("agent_error", f"I hit a snag talking to my brain: {e.__class__.__name__}")
        return parse_reply(raw)

    def tool_result(self, reply: CompanionReply) -> CompanionReply:
        """Tools return CompanionReplies directly; normalize defensively."""
        return parse_reply(reply)

    # ------------------------------------------------------------------
    # convenience for the two "fast lane" patterns
    # ------------------------------------------------------------------

    @property
    def fast(self) -> LLMClient:
        return self.llm  # call with model=self.fast_model for structured work


def create_companion_agent(
    system_prompt: str,
    tools: Optional[List[Callable]] = None,
    name: str = "companion",
    model: str = DEFAULT_REASONING_MODEL,
    fast_model: str = DEFAULT_FAST_MODEL,
    llm: Optional[LLMClient] = None,
    **agent_kwargs: Any,
) -> CompanionAgent:
    """One-call construction. See README.md quickstart."""
    return CompanionAgent(
        name=name,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        fast_model=fast_model,
        llm=llm,
        **agent_kwargs,
    )
