"""Response protocol.

TV clients must never have to guess what the model said. Every reply from a
companion agent is validated into `CompanionReply` so the display layer can
render a small fixed set of card types. If the model returns something
unparseable, the kit downgrades it to `kind="answer"` with the raw text
instead of crashing the app.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CompanionAction(str, Enum):
    """Card types a TV client knows how to render."""

    BRIEFING = "briefing"
    ANSWER = "answer"
    QUIZ = "quiz"
    QUIZ_RESULT = "quiz_result"
    EXPLAIN = "explain"
    DISCUSSION = "discussion"
    PROGRESS_UPDATE = "progress_update"
    ERROR = "error"


class QuizQuestion(BaseModel):
    id: str = "q1"
    question: str
    options: List[str] = Field(min_length=2)
    correct_index: int
    explanation: str = ""

    @field_validator("correct_index")
    @classmethod
    def _in_range(cls, v: int, info: Any) -> int:
        # options length isn't visible here; validated by Quiz below
        return v


class Quiz(BaseModel):
    topic: str
    difficulty: str = "medium"
    questions: List[QuizQuestion] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def _check_indices(cls, v: List[QuizQuestion]) -> List[QuizQuestion]:
        for i, q in enumerate(v):
            if not (0 <= q.correct_index < len(q.options)):
                raise ValueError(
                    f"question {i}: correct_index {q.correct_index} out of range "
                    f"for {len(q.options)} options"
                )
            if not q.id:
                q.id = f"q{i + 1}"
        return v


class ProgressUpdate(BaseModel):
    title: str
    season: int = Field(ge=0)
    episode: int = Field(ge=0)
    note: str = ""


class BriefingSection(BaseModel):
    heading: str
    body: str


class CompanionReply(BaseModel):
    """The single shape a TV client ever receives."""

    kind: CompanionAction = CompanionAction.ANSWER
    text: str = ""
    # Optional structured payloads, present only for matching kinds:
    quiz: Optional[Quiz] = None
    quiz_result: Optional[Dict[str, Any]] = None  # {score, total, correct: [...], review: [...]}
    progress: Optional[ProgressUpdate] = None
    sections: Optional[List[BriefingSection]] = None
    level: Optional[str] = None  # explain mode: "kid" | "standard" | "deep"
    spoiler_warning: Optional[str] = None  # human-readable reason, if a guard fired
    error_code: Optional[str] = None

    @classmethod
    def from_unstructured(cls, raw: str) -> "CompanionReply":
        """Downgrade path: keep the TV alive when the model misbehaves."""
        text = (raw or "").strip()
        return cls(kind=CompanionAction.ANSWER, text=text or "Sorry, I couldn't get that answer. Try again?")

    @classmethod
    def error(cls, code: str, message: str) -> "CompanionReply":
        return cls(kind=CompanionAction.ERROR, text=message, error_code=code)

    def to_wire(self) -> Dict[str, Any]:
        """JSON-safe dict for the /chat response body."""
        data = self.model_dump(mode="json", exclude_none=True)
        data["kind"] = self.kind.value
        return data


def parse_reply(value: Any) -> CompanionReply:
    """Coerce whatever the model produced into a CompanionReply.

    Accepts: CompanionReply | dict | JSON str | plain str. Never raises.
    """
    if isinstance(value, CompanionReply):
        return value
    if isinstance(value, dict):
        try:
            return CompanionReply.model_validate(value)
        except Exception:
            return CompanionReply.from_unstructured(str(value))
    if isinstance(value, str):
        s = value.strip()
        # Strip common code fences before parsing.
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        if s.startswith("{"):
            import json

            try:
                return CompanionReply.model_validate(json.loads(s))
            except Exception:
                pass
        return CompanionReply.from_unstructured(value)
    return CompanionReply.from_unstructured(str(value))
