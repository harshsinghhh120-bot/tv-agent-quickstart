"""LLM access with an injectable client.

Every model call in the kit flows through `LLMClient.complete()`:

* `BedrockLLMClient` — Amazon Bedrock `converse` via boto3 (the default
  for AWS deployments).
* `FakeLLMClient` — a callable-driven stub for tests and offline dev.

The default models follow the companion pattern: a fast, cheap model
(Haiku) for latency-sensitive structured work (quizzes, explanations,
grading) and a strong model (Sonnet) for open-ended companion turns.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional, Protocol

# Sensible defaults; override per-region / per-account in create_companion_agent().
DEFAULT_REASONING_MODEL = "anthropic.claude-sonnet-4-5"
DEFAULT_FAST_MODEL = "anthropic.claude-haiku-4-5"


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        json_mode: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Return the model's text output."""
        ...


class BedrockLLMClient:
    """boto3-backed Bedrock converse client (lazy import)."""

    def __init__(self, region: Optional[str] = None, profile: Optional[str] = None):
        self.region = region
        self.profile = profile
        self._client = None

    def _boto(self):
        if self._client is None:
            import boto3  # lazy: keeps the package importable without AWS

            kwargs: Dict[str, Any] = {}
            if self.region:
                kwargs["region_name"] = self.region
            if self.profile:
                kwargs["profile_name"] = self.profile
            self._client = boto3.client("bedrock", **kwargs)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        json_mode: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        model_id = model or (DEFAULT_FAST_MODEL if json_mode else DEFAULT_REASONING_MODEL)
        system_block = [{"text": system}] if system else []
        body: Dict[str, Any] = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_block:
            body["system"] = system_block
        if json_mode:
            body["responseFormat"] = {"type": "json"}  # honored by Claude on Bedrock
        resp = self._boto().converse(**body)
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


class FakeLLMClient:
    """Test/offline stub. `fn` receives (system, user, kwargs) and returns text.

    A *registry* is supported so tests can assert which prompt ran:

        calls = []
        fake = FakeLLMClient(lambda s, u, k: "..." , record=calls)
    """

    def __init__(self, fn: Callable[..., str], record: Optional[list] = None):
        self.fn = fn
        self.calls = record if record is not None else []

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.fn(system, user, kwargs)


def parse_json_block(raw: str) -> Dict[str, Any]:
    """Parse a JSON object out of model output, tolerating code fences and
    leading prose. Raises ValueError if nothing parseable is found."""
    if raw is None:
        raise ValueError("empty model output")
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    elif s.startswith("{"):
        # Greedy scan for the outermost braces.
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    s = s[: i + 1]
                    break
    else:
        # No leading brace: find the first {...} block.
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object found in output: {raw[:120]!r}")
        s = m.group(0)
    return json.loads(s)


def complete_json(client: LLMClient, system: str, user: str, **kwargs: Any) -> Dict[str, Any]:
    """complete() + robust JSON parse. Retries once on parse failure."""
    out = client.complete(system, user, json_mode=True, **kwargs)
    try:
        return parse_json_block(out)
    except (ValueError, json.JSONDecodeError):
        retry = client.complete(
            system + "\n\nIMPORTANT: Your previous output was not valid JSON. Respond with ONLY a JSON object, no prose.",
            user,
            json_mode=True,
            **kwargs,
        )
        return parse_json_block(retry)
