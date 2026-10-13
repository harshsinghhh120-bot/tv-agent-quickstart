"""HTTP surface for a companion agent.

`create_app()` returns a FastAPI app with:

  GET  /health  -> liveness + agent name
  POST /chat    -> one companion turn

Run locally:   uvicorn.run(create_app(agent, memory), host="0.0.0.0", port=8000)
Deploy:        wrap the same app in Lambda, or use agentcore.py to run the
               agent on Amazon Bedrock AgentCore instead.

Wire format:

  POST /chat
  {
    "viewer_id": "couch-1",
    "message": "Why did she do that?",
    "action":  null | "quiz_answer" | ... ,
    "payload": { ... }              # action-specific structured input
  }
  -> 200  CompanionReply JSON
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .agent import CompanionAgent
from .memory import MemoryStore
from .protocol import CompanionReply


class ChatRequest(BaseModel):
    viewer_id: str = "default"
    message: str = ""
    action: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


def _action_to_message(req: ChatRequest) -> str:
    """Fold structured client actions into a natural-language turn so the
    agent (not the server) stays in charge of conversation logic."""
    msg = req.message or ""
    if req.action == "quiz_answer" and req.payload:
        qid = req.payload.get("question_id", "?")
        choice = req.payload.get("choice", "")
        msg = f"[structured] The viewer answered question {qid} with: {choice}"
    elif req.action == "select" and req.payload:
        msg = f"[structured] The viewer selected: {req.payload.get('value', '')}"
    return msg


def create_app(
    agent: CompanionAgent,
    memory: MemoryStore,
    api_key: Optional[str] = None,
    title: str = "TV Companion Agent",
) -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The HTTP server needs FastAPI: pip install 'tv-agent-quickstart[serve]'"
        ) from e

    api = api_key if api_key is not None else os.environ.get("COMPANION_API_KEY")
    app = FastAPI(title=title)

    def _auth(x_api_key: Optional[str] = Header(default=None)):
        if api and x_api_key != api:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "agent": agent.name}

    @app.post("/chat", dependencies=[Depends(_auth)] if api else [])
    def chat(req: ChatRequest) -> Dict[str, Any]:
        state = memory.get(req.viewer_id)
        reply = agent.ask(
            viewer_id=req.viewer_id,
            message=_action_to_message(req),
            state=state,
        )
        # Protocol-driven memory sync: any card carrying a progress record is
        # persisted, so viewer state stays consistent whether the model called
        # a tool or simply returned a progress_update card.
        if reply.progress is not None:
            try:
                memory.set_progress(
                    req.viewer_id,
                    reply.progress.title,
                    reply.progress.season,
                    reply.progress.episode,
                    note=reply.progress.note or "",
                )
            except Exception:  # noqa: BLE001 — memory must never break the turn
                pass
        # Tools that mutate state have already written to `memory` during the
        # turn (they close over the same store the server passes around).
        if reply.kind.value == "error" and reply.error_code in ("no_agent",):
            from fastapi import status as _s

            return _json_error(409, reply)
        return _ok(reply)

    def _ok(reply: CompanionReply) -> Dict[str, Any]:
        return reply.to_wire()

    def _json_error(status_code: int, reply: CompanionReply) -> Any:
        from fastapi import JSONResponse

        return JSONResponse(status_code=status_code, content=reply.to_wire())

    return app
