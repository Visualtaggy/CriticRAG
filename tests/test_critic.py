"""
Tests for CriticAgent — all Ollama calls are mocked.
"""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest

from criticrag.agents.critic import CriticAgent

HITS = [
    (0.9, "aurora", "Solar wind particles collide with atmospheric gases causing the aurora."),
    (0.7, "magnetism", "Earth's magnetic field deflects most solar wind particles."),
]
QUERY = "What causes aurora borealis?"

def _mock_judge(scores: dict, failure_type: str = "incomplete") -> str:
    """Return a fake judge response for the scoring call."""
    return json.dumps({**scores, "reasoning": "test reasoning"})

def _mock_judge_fail(scores: dict, failure_type: str) -> str:
    return json.dumps({
        "failure_type": failure_type,
        "failure_reason": "Test failure reason.",
    })

class TestCriticAgent:
    def setup_method(self):
        self.critic = CriticAgent()

    def test_passes_on_high_scores(self):
        good_scores = {"faithfulness": 8, "relevance": 9, "completeness": 7}
        with mock.patch("criticrag.agents.critic.judge", return_value=_mock_judge(good_scores)):
            result = self.critic.evaluate(QUERY, HITS, "Solar wind causes aurora.")
        assert result.passed is True
        assert result.failure_type is None
        assert result.mean_score > 6.5

    def test_fails_on_low_scores(self):
        bad_scores = {"faithfulness": 2, "relevance": 3, "completeness": 1}
        fail_response = json.dumps({"failure_type": "hallucination", "failure_reason": "Made up facts."})

        call_count = 0
        def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({**bad_scores, "reasoning": "low quality"})
            return fail_response

        with mock.patch("criticrag.agents.critic.judge", side_effect=side_effect):
            result = self.critic.evaluate(QUERY, HITS, "Some hallucinated answer.")

        assert result.passed is False
        assert result.failure_type == "hallucination"
        assert result.mean_score < 6.5

    def test_degrades_gracefully_on_bad_json(self):
        with mock.patch("criticrag.agents.critic.judge", return_value="not json at all !!!"):
            result = self.critic.evaluate(QUERY, HITS, "Some answer.")
        assert result.mean_score == 0.0
        assert result.passed is False

    def test_failure_type_in_taxonomy(self):
        bad_scores = {"faithfulness": 1, "relevance": 1, "completeness": 1}
        fail_response = json.dumps({"failure_type": "no_docs", "failure_reason": "Nothing relevant."})

        call_count = 0
        def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            return json.dumps({**bad_scores, "reasoning": "x"}) if call_count == 1 else fail_response

        with mock.patch("criticrag.agents.critic.judge", side_effect=side_effect):
            result = self.critic.evaluate(QUERY, HITS, "I don't know.")

        assert result.failure_type in ("no_docs", "hallucination", "incomplete")
