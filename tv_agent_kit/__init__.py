"""tv-agent-quickstart — scaffold an AI companion agent for TV apps.

Public API. See README.md for the 10-minute quickstart.
"""

from .protocol import (
    CompanionAction,
    CompanionReply,
    parse_reply,
    ProgressUpdate,
    Quiz,
    QuizQuestion,
    BriefingSection,
)
from .memory import MemoryStore, LocalMemoryStore, AgentCoreMemoryStore
from .llm import (
    DEFAULT_FAST_MODEL,
    DEFAULT_REASONING_MODEL,
    BedrockLLMClient,
    FakeLLMClient,
    LLMClient,
    complete_json,
    parse_json_block,
)
from .agent import CompanionAgent, create_companion_agent, tool
from .server import create_app

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FAST_MODEL",
    "DEFAULT_REASONING_MODEL",
    "CompanionAction",
    "parse_reply",
    "CompanionReply",
    "ProgressUpdate",
    "Quiz",
    "QuizQuestion",
    "BriefingSection",
    "MemoryStore",
    "LocalMemoryStore",
    "AgentCoreMemoryStore",
    "LLMClient",
    "BedrockLLMClient",
    "FakeLLMClient",
    "complete_json",
    "parse_json_block",
    "CompanionAgent",
    "create_companion_agent",
    "tool",
    "create_app",
    "__version__",
]
