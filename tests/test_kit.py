"""Offline unit tests for tv-agent-quickstart (no AWS, no network)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tv_agent_kit import (  # noqa: E402
    CompanionAction,
    CompanionReply,
    FakeLLMClient,
    LocalMemoryStore,
    create_companion_agent,
    parse_json_block,
)
from tv_agent_kit.protocol import Quiz, QuizQuestion, parse_reply  # noqa: E402


class TestProtocol(unittest.TestCase):
    def test_plain_text_downgrades_to_answer(self):
        r = parse_reply("Sure, here's the story...")
        self.assertEqual(r.kind, CompanionAction.ANSWER)
        self.assertIn("story", r.text)

    def test_json_string_parses(self):
        r = parse_reply('{"kind": "quiz", "quiz": {"topic": "t", "questions": '
                        '[{"question": "q", "options": ["a", "b"], "correct_index": 0}]}}')
        self.assertEqual(r.kind, CompanionAction.QUIZ)
        self.assertEqual(r.quiz.topic, "t")

    def test_code_fenced_json_parses(self):
        r = parse_reply('```json\n{"kind": "explain", "level": "kid", "text": "hi"}\n```')
        self.assertEqual(r.kind, CompanionAction.EXPLAIN)
        self.assertEqual(r.level, "kid")

    def test_broken_json_falls_back(self):
        r = parse_reply('{"kind": "quiz", "quiz": oops')
        self.assertEqual(r.kind, CompanionAction.ANSWER)

    def test_bad_correct_index_rejected(self):
        with self.assertRaises(Exception):
            Quiz(
                topic="t",
                questions=[QuizQuestion(question="q", options=["a", "b"], correct_index=5)],
            )

    def test_to_wire_is_json_safe(self):
        r = CompanionReply(kind=CompanionAction.PROGRESS_UPDATE,
                           text="ok",
                           progress={"title": "X", "season": 1, "episode": 2})
        wire = json.loads(json.dumps(r.to_wire()))
        self.assertEqual(wire["kind"], "progress_update")


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.store = LocalMemoryStore(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_roundtrip(self):
        self.store.update("v1", mood="chill")
        self.assertEqual(self.store.get("v1")["mood"], "chill")
        self.assertEqual(self.store.get("v2"), {})

    def test_progress_helpers(self):
        self.store.set_progress("v1", " The Bear ", 2, 8, "finale")
        p = self.store.get_progress("v1", "the bear")
        self.assertEqual((p["season"], p["episode"]), (2, 8))
        self.assertEqual(p["note"], "finale")

    def test_quiz_score(self):
        self.store.bump_quiz_score("v1", 2, 3)
        self.store.bump_quiz_score("v1", 1, 3)
        q = self.store.get("v1")["quizzes"]
        self.assertEqual((q["games"], q["correct"], q["answered"]), (2, 3, 6))

    def test_corrupt_file_recovers(self):
        with open(self.tmp.name, "w") as f:
            f.write("{not json")
        self.assertEqual(self.store.get("v1"), {})


class TestLLM(unittest.TestCase):
    def test_parse_json_block_plain(self):
        self.assertEqual(parse_json_block('{"a": 1}'), {"a": 1})

    def test_parse_json_block_fenced(self):
        self.assertEqual(parse_json_block('blah\n```json\n{"a": [1, 2]}\n```\ntrailing'), {"a": [1, 2]})

    def test_parse_json_block_prose_prefix(self):
        self.assertEqual(parse_json_block('Here you go: {"a": {"b": 2}} done'), {"a": {"b": 2}})

    def test_parse_json_block_garbage_raises(self):
        with self.assertRaises(ValueError):
            parse_json_block("no json here")

    def test_fake_client_records_calls(self):
        calls = []
        fake = FakeLLMClient(lambda s, u, k: "hi", record=calls)
        self.assertEqual(fake.complete("sys", "usr", json_mode=True), "hi")
        self.assertTrue(calls and calls[0]["json_mode"] is True)


class TestAgent(unittest.TestCase):
    def test_ask_without_backend_returns_error_reply(self):
        agent = create_companion_agent(system_prompt="p", name="t")
        if agent._agent is None:
            r = agent.ask("v1", "hello", state={})
            self.assertEqual(r.kind, CompanionAction.ERROR)
            self.assertEqual(r.error_code, "no_agent")
        else:
            self.skipTest("strands installed; no_agent path not exercised")

    def test_context_block_includes_state(self):
        agent = create_companion_agent(system_prompt="p")
        block = agent.context_block("v1", {"progress": {"the bear": {"season": 2, "episode": 8}}})
        self.assertIn("v1", block)
        self.assertIn("the bear", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
