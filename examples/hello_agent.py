"""10-minute hello-world companion.

Run:
    python examples/hello_agent.py

With AWS configured (bedrock extra installed) this builds a real Strands
agent on Bedrock and prints one turn. Without AWS it falls back to a stub
LLM so you can see the *shape* of the kit offline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tv_agent_kit import (  # noqa: E402
    CompanionReply,
    FakeLLMClient,
    LocalMemoryStore,
    create_companion_agent,
    tool,
)

SYSTEM_PROMPT = """You are a friendly TV viewing companion on a living-room
display. Keep answers short (max 3 sentences) — it's a 10-foot display.
If a viewer tells you how far they are in a show, NEVER spoil anything
after their point. Reply in JSON matching the CompanionReply protocol."""


@tool
def set_progress(viewer: str, title: str, season: int, episode: int) -> CompanionReply:
    """Record how far a viewer is in a show so future answers avoid spoilers.

    viewer: viewer id, title: show title, season/episode: progress position.
    """
    return CompanionReply(
        kind="progress_update",
        text=f"Got it — you're on {title} S{season}E{episode}. I'll keep spoilers locked.",
        progress={"title": title, "season": season, "episode": episode},
    )


def _stub_llm(system: str, user: str, kwargs: dict) -> str:
    """Offline stand-in so this example runs without AWS credentials."""
    return (
        '{"kind": "answer", "text": "Hello from the offline stub! '
        "Configure AWS (pip install tv-agent-quickstart[bedrock]) for real answers."}'
    )


def main() -> None:
    memory = LocalMemoryStore("hello-state.json")

    has_aws = bool(os.environ.get("AWS_ACCESS_KEY_ID")) and _strands_available()
    agent = create_companion_agent(
        system_prompt=SYSTEM_PROMPT,
        tools=[set_progress],
        name="hello-companion",
        llm=None if has_aws else FakeLLMClient(_stub_llm),
    )
    if not has_aws and agent._agent is None:
        print("NOTE: strands-agents not installed — running offline with a stub agent.")

    viewer = "couch-1"
    state = memory.get(viewer)
    reply = agent.ask(viewer_id=viewer, message="Hi! What can you do?", state=state)
    print(f"-> {reply.kind.value}: {reply.text}")
    print("Wire JSON:", reply.to_wire())


def _strands_available() -> bool:
    try:
        import strands  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
