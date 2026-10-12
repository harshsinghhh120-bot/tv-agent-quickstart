# tv-agent-quickstart

**Scaffold an AI companion agent for TV apps in ~10 minutes.**

`tv-agent-quickstart` is a small, dependency-light Python toolkit for building
voice-and-remote-friendly AI companion agents for living-room displays
(Fire TV, smart TVs, set-top boxes). It was extracted from the
[ReelMind](https://github.com/<YOUR_GITHUB>/reelmind) hackathon project
(Build, Ship, Shape: Amazon Developer Hackathon, Oct 2026).

## What you get

| Piece | What it does |
|---|---|
| `create_companion_agent()` | One-call construction of a Strands (AWS) agent on Amazon Bedrock with a companion system prompt and your tools |
| `bedrock_converse()` | Thin, dependency-light helper for direct Bedrock `converse` calls (Haiku for fast structured work, Sonnet for reasoning) |
| `MemoryStore` | Per-viewer persistent state. Ships with a zero-dependency JSON-file store; add an AgentCore Memory backend behind the same interface |
| `CompanionReply` | A pydantic response protocol so TV clients always get predictable JSON (`briefing` / `answer` / `quiz` / `quiz_result` / `explain` / `discussion` / `progress_update` / `error`) |
| `create_app()` | A FastAPI server exposing `POST /chat` and `GET /health` — run locally, on Lambda, or deploy to Amazon Bedrock AgentCore |
| `agentcore.py` | Optional wrapper that runs the same agent on Amazon Bedrock AgentCore (runtime + memory + gateway) |

## Why it exists

Building a "chat box" is easy; building a *companion* for a 10-foot display
is not. TV apps need: predictable response shapes (no free-text surprises),
per-viewer memory (who's on the couch, how far along are they), spoiler/
safety policies that survive model drift, and a cheap "fast lane" model for
latency-sensitive turns. This kit encodes those patterns so the next TV
companion starts from a working shape instead of a blank notebook.

## Quickstart

```bash
pip install tv-agent-quickstart   # or: pip install git+https://github.com/<you>/tv-agent-quickstart.git
```

### 1. Define your tools (pure Python, testable without AWS)

```python
from tv_agent_kit import tool, CompanionAction

@tool
def set_progress(viewer: str, title: str, season: int, episode: int) -> str:
    """Record how far a viewer is in a show so answers can avoid spoilers."""
    ...
```

### 2. Build the agent

```python
from tv_agent_kit import create_companion_agent

agent = create_companion_agent(
    system_prompt=open("system_prompt.txt").read(),
    tools=[set_progress],
    model="anthropic.claude-sonnet-4-5",          # reasoning turns
)
reply = agent.ask(viewer_id="couch-1", message="Why did she do that?")
print(reply.kind, reply.text)                     # structured, always
```

### 3. Serve it to your TV app

```python
import uvicorn
from tv_agent_kit import create_app, LocalMemoryStore

app = create_app(agent, LocalMemoryStore("state.json"))
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Your TV app then calls:

```
POST /chat
{"viewer_id": "couch-1", "message": "Pop quiz, 3 questions, hard"}
-> 200 {"kind": "quiz", "quiz": {"topic": "...", "questions": [...]}, ...}
```

## Repository layout

```
tv_agent_kit/
├── __init__.py      # public API
├── agent.py         # create_companion_agent, CompanionAgent
├── llm.py           # bedrock_converse() + injectable LLM client (tests run offline)
├── memory.py        # MemoryStore protocol, LocalMemoryStore, AgentCoreMemoryStore
├── protocol.py      # CompanionReply + CompanionAction enum + payload models
├── server.py        # FastAPI app factory, /chat + /health, optional API key
└── agentcore.py     # optional Amazon Bedrock AgentCore deployment wrapper
examples/
└── hello_agent.py   # 10-minute hello-world companion
tests/
└── test_kit.py      # offline unit tests (stubbed LLM)
```

## Design notes

- **LLM is injectable.** Every model call flows through an `LLMClient`
  interface. The default is Bedrock via boto3; tests (and your own
  environments) can inject a stub. The core logic — memory, protocol,
  quiz grading policy, progress math — runs and tests with **no AWS
  account at all**.
- **The model decides, the kit enforces.** The companion system prompt
  tells the model the policy (spoilers, tone, output shape); the kit
  validates the final reply against the pydantic protocol and downgrades
  malformed output to `kind="answer"` rather than crashing the TV app.
- **Memory is a strategy object.** `LocalMemoryStore` is a JSON file so
  the quickstart works on a laptop. `AgentCoreMemoryStore` implements the
  same interface on Amazon Bedrock AgentCore Memory when you deploy to
  AWS. Your TV code never knows the difference.

## Relationship to ReelMind

`ReelMind` (the hackathon submission) `pip install`s this package and
supplies its own tools, system prompt, and catalog. Everything in this
repo is reusable without ReelMind; everything ReelMind does on the
backend runs on this kit at runtime.

## License

Apache-2.0 — see [LICENSE](LICENSE).
