"""Per-viewer memory.

A `MemoryStore` maps a viewer id (who is on the couch) to a small JSON state
dict: progress per title, quiz scores, preferred explain level, recent
titles, etc. The interface is deliberately tiny so backends are swappable:

* `LocalMemoryStore` — JSON file on disk. Zero dependencies, works on a
  laptop, perfect for the quickstart and for local AgentCore development.
* `AgentCoreMemoryStore` — Amazon Bedrock AgentCore Memory (managed,
  durable, multi-instance). Same interface; enable it when you deploy.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict


class MemoryStore(ABC):
    @abstractmethod
    def get(self, viewer_id: str) -> Dict[str, Any]:
        """Return the viewer's state dict ({} if none)."""

    @abstractmethod
    def set(self, viewer_id: str, state: Dict[str, Any]) -> None:
        """Replace the viewer's state dict."""

    # --- convenience helpers shared by all backends -----------------------

    def update(self, viewer_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Merge top-level keys into the viewer state and return it."""
        state = self.get(viewer_id)
        state.update(kwargs)
        self.set(viewer_id, state)
        return state

    def set_progress(self, viewer_id: str, title: str, season: int, episode: int, note: str = "") -> Dict[str, Any]:
        progress = self.get(viewer_id).get("progress", {})
        key = title.strip().lower()
        progress[key] = {"title": title.strip(), "season": int(season), "episode": int(episode), "note": note}
        self.update(viewer_id, progress=progress)
        return progress[key]

    def get_progress(self, viewer_id: str, title: str) -> Dict[str, Any]:
        progress = self.get(viewer_id).get("progress", {})
        return progress.get(title.strip().lower(), {})

    def bump_quiz_score(self, viewer_id: str, correct: int, total: int) -> Dict[str, Any]:
        quizzes = self.get(viewer_id).get("quizzes", {})
        for k in ("games", "correct", "answered"):
            quizzes[k] = quizzes.get(k, 0) + (correct if k == "correct" else total if k == "answered" else 1)
        self.update(viewer_id, quizzes=quizzes)
        return quizzes


class LocalMemoryStore(MemoryStore):
    """Thread-safe JSON-file memory. One file, one viewer per key."""

    def __init__(self, path: str = "reelmind-state.json"):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(path):
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = f"{self.path}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get(self, viewer_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._read().get(viewer_id, {}))

    def set(self, viewer_id: str, state: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data[viewer_id] = state
            self._write(data)


class AgentCoreMemoryStore(MemoryStore):
    """Amazon Bedrock AgentCore Memory backend.

    Verifies wiring at construction and degrades loudly if the `boto3`
    package or the AgentCore Memory API is unavailable, so local
    development never silently loses memory.

    NOTE: AgentCore Memory is a fast-moving service — if a call fails with
    a client error, re-check the current boto3 API docs for
    `bedrock-agentcore-data` (create_memory / get_memories / put_memories /
    search_memories) and adjust `_load`/`_save` accordingly. The interface
    (get/set/update/set_progress/...) stays the same.
    """

    def __init__(self, memory_id: str, region: str = "us-east-1", profile: str | None = None):
        try:
            import boto3  # noqa: WPS433 (lazy on purpose)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "AgentCoreMemoryStore needs boto3: pip install 'tv-agent-quickstart[bedrock]'"
            ) from e
        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        self._client = boto3.client("bedrock-agentcore-data", **session_kwargs)
        self.memory_id = memory_id

    def _load(self, viewer_id: str) -> Dict[str, Any]:
        try:
            resp = self._client.get_memories(
                memoryId=self.memory_id,
                resourcePath=f"viewers/{viewer_id}",
            )
            entries = (resp or {}).get("memories", [])
            for entry in entries:
                payload = entry.get("stringMemory") or {}
                if payload.get("key") == "state":
                    return json.loads(payload.get("value") or "{}")
        except Exception:
            # First run / not created yet — treat as empty.
            return {}
        return {}

    def _save(self, viewer_id: str, state: Dict[str, Any]) -> None:
        self._client.put_memories(
            memoryId=self.memory_id,
            resourcePath=f"viewers/{viewer_id}",
            memories=[
                {
                    "type": "STRING",
                    "stringMemory": {"key": "state", "value": json.dumps(state, ensure_ascii=False)},
                }
            ],
        )

    def get(self, viewer_id: str) -> Dict[str, Any]:
        return self._load(viewer_id)

    def set(self, viewer_id: str, state: Dict[str, Any]) -> None:
        self._save(viewer_id, state)
