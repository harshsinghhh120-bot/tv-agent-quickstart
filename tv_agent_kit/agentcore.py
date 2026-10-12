"""Optional Amazon Bedrock AgentCore deployment wrapper.

AgentCore gives a Strands agent a managed runtime, memory, gateway
(REST/MCP endpoint), identity, and observability. The wrapper is lazy so
the kit (and its tests) work without the `agentcore` package installed.

Usage (see reelmind/agent README for the full deploy walk-through):

    from tv_agent_kit import create_companion_agent
    from tv_agent_kit.agentcore import to_agentcore

    agent = create_companion_agent(system_prompt=..., tools=[...])
    runtime = to_agentcore(agent, name="reelmind-companion")
    runtime.dev()      # local dev server with AgentCore tooling
    runtime.deploy()   # push to your AWS account

If the `agentcore` package API has moved since this was written, the
error message points you at the current docs instead of failing silently.
"""

from __future__ import annotations

from typing import Any, Optional

from .agent import CompanionAgent


class AgentCoreRuntime:
    """Thin adapter over the official `agentcore` Strands integration."""

    def __init__(self, agent: CompanionAgent, name: str, region: Optional[str] = None):
        try:
            from agentcore import AgentCoreAgent  # lazy on purpose
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Amazon Bedrock AgentCore support needs the 'agentcore' package:\n"
                "  pip install 'tv-agent-quickstart[agentcore]'\n"
                "Then set AWS credentials (aws configure) and a default region. "
                "Docs: https://docs.aws.amazon.com/bedrock-agentcore/"
            ) from e
        self._core = AgentCoreAgent(agent._agent, name=name)
        self.region = region

    def dev(self, **kwargs: Any) -> None:  # pragma: no cover (network)
        """Run the local AgentCore dev loop."""
        self._core.dev(**kwargs)

    def deploy(self, **kwargs: Any) -> Any:  # pragma: no cover (network)
        """Deploy to AgentCore; returns the deployed runtime handle."""
        return self._core.deploy(**kwargs)


def to_agentcore(agent: CompanionAgent, name: str, region: Optional[str] = None) -> AgentCoreRuntime:
    return AgentCoreRuntime(agent, name=name, region=region)
