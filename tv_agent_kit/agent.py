"""The companion agent.

`create_companion_agent()` builds a Strands (AWS) agent on Amazon Bedrock
with a companion system prompt and your tools. `CompanionAgent.ask()` is
the single entry point the server (and your TV app) uses, and it *always*
returns a validated `CompanionReply`.

Two execution modes:

* **agent mode** (default): a real Strands agent loop on Bedrock — the model
  can call your tools (progress, quiz, explain, ...).
* **simple mode** (`simple=True`): model-direct — one `LLMClient.complete()`
  call per turn, no agent loop, no tools. Perfect for demos/offline dev
  (pair with a stub LLMClient) and as a cheap fallback. Conversation context
  must be included in the prompt by the caller (the kit injects stored
  viewer state; the client app should include the show title in messages).

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
    passes through unchanged so unit tests can run anywhere. The decorated
    result (strands' `DecoratedFunctionTool`) stays directly callable.
    """
    if _HAS_STRANDS:
        return _strands_tool(fn)
    return fn


def _extract_text(result: Any) -> str:
    """Coerce an agent/model result into text.

    Handles: str | dict (message or AgentResult.to_dict shape) | AgentResult
    (with .message) | anything else (str()).
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result

    def _text_blocks(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        out = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                out.append(str(item.get("text", "")))
            elif isinstance(item, str):
                out.append(item)
        return "\n".join(out)

    if isinstance(result, dict):
        if "content" in result:
            return _text_blocks(result.get("content"))
        inner = result.get("message")
        if isinstance(inner, dict):
            return _text_blocks(inner.get("content"))
        return json.dumps(result)
    so = getattr(result, "structured_output", None)
    if so is not None:
        try:
            return so.model_dump_json()
        except Exception:  # noqa: BLE001
            return str(so)
    msg = getattr(result, "message", None)
    if isinstance(msg, dict):
        return _text_blocks(msg.get("content"))
    return str(result)


class CompanionAgent:
    """A companion with a guaranteed reply contract."""

    def __init__(
        self,
        name: str = "companion",
        system_prompt: str = "",
        tools: Optional[List[Callable]] = None,
        model: str = DEFAULT_REASONING_MODEL,
        fast_model: str = DEFAULT_FAST_MODEL,
        llm: Optional[LLMClient] = None,
        strands_agent: Optional[Any] = None,
        simple: bool = False,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.fast_model = fast_model
        self.simple = simple
        self.llm: LLMClient = llm or BedrockLLMClient()
        self._tools = tools or []

        if simple:
            # Model-direct mode: no agent loop, no tools.
            self._agent = None
        elif strands_agent is not None:
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
        if self.simple:
            prompt = f"{self.context_block(viewer_id, state or {})}\n\nUser message: {message}"
            try:
                raw = self.llm.complete(self.system_prompt, prompt, **kwargs)
            except Exception as e:  # noqa: BLE001 — the TV must never crash
                return CompanionReply.error("llm_error", f"Model call failed: {e.__class__.__name__}")
            return parse_reply(raw)

        if self._agent is None:
            return CompanionReply.error(
                "no_agent",
                "The agent backend is not available. Install 'tv-agent-quickstart[bedrock]' "
                "(strands-agents + boto3) and configure AWS credentials — or run in DEMO_MODE "
                "(simple mode with a stub LLM) for offline development.",
            )
        prompt = (
            f"{self.context_block(viewer_id, state or {})}\n\n"
            f"User message: {message}"
        )
        try:
            result = self._agent(prompt, **kwargs)
        except Exception as e:  # noqa: BLE001 — the TV must never crash
            return CompanionReply.error("agent_error", f"I hit a snag talking to my brain: {e.__class__.__name__}")
        return parse_reply(_extract_text(result))

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
    simple: bool = False,
    **agent_kwargs: Any,
) -> CompanionAgent:
    """One-call construction. See README.md quickstart.

    `simple=True` → model-direct mode (no Strands agent loop); the injected
    `llm` is the only model dependency. Ideal for demos and offline dev.
    """
    return CompanionAgent(
        name=name,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        fast_model=fast_model,
        llm=llm,
        simple=simple,
        **agent_kwargs,
    )
