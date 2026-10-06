"""
Integration tests for CriticRAGPipeline and NaiveRAGPipeline.
All LLM calls are mocked — tests run without Ollama.
"""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest

from criticrag.pipeline import CriticRAGPipeline, NaiveRAGPipeline
from criticrag.agents.router import RouterAgent
from criticrag.agents.generator import GeneratorAgent
from criticrag.agents.critic import CriticAgent
from criticrag.retrieval.sparse import SparseIndex
from criticrag.retrieval.hybrid import HybridIndex

CHUNKS = [
    ("aurora", "Solar wind particles interact with Earth's magnetic field causing auroras."),
    ("photosyn", "Plants convert sunlight and CO2 into glucose via photosynthesis."),
]
QUERY = "What causes aurora borealis?"

def _build_pipeline(pass_immediately=True):
    sparse = SparseIndex(CHUNKS)

    dense_mock = mock.MagicMock()
    dense_mock.search.return_value = [(0.9, "aurora", CHUNKS[0][1])]
    dense_mock.chunks = CHUNKS

    hybrid_mock = mock.MagicMock()
    hybrid_mock.search.return_value = [(0.9, "aurora", CHUNKS[0][1])]

    router = RouterAgent(dense_mock, sparse, hybrid_mock)

    generator = mock.MagicMock(spec=GeneratorAgent)
    generator.answer.return_value = "Solar wind causes aurora borealis."

    critic = mock.MagicMock(spec=CriticAgent)
    from criticrag.agents.critic import CriticResult
    if pass_immediately:
        critic.evaluate.return_value = CriticResult(
            scores={"faithfulness": 9.0, "relevance": 9.0, "completeness": 8.0},
            mean_score=8.67,
            passed=True,
            failure_type=None,
            failure_reason=None,
            reasoning="Good answer.",
        )
    else:
        fail = CriticResult(
            scores={"faithfulness": 2.0, "relevance": 2.0, "completeness": 2.0},
            mean_score=2.0,
            passed=False,
            failure_type="incomplete",
            failure_reason="Answer is too vague.",
            reasoning="Incomplete.",
        )
        success = CriticResult(
            scores={"faithfulness": 8.0, "relevance": 8.0, "completeness": 8.0},
            mean_score=8.0,
            passed=True,
            failure_type=None,
            failure_reason=None,
            reasoning="Good.",
        )
        critic.evaluate.side_effect = [fail, success]

    return CriticRAGPipeline(router, generator, critic), router, generator, critic

class TestCriticRAGPipeline:
    def test_returns_answer_on_first_pass(self):
        pipeline, _, gen, _ = _build_pipeline(pass_immediately=True)
        with mock.patch("criticrag.utils.tracer.LOG_TRACES", False):
            answer, trace = pipeline.run(QUERY)
        assert "solar wind" in answer.lower() or len(answer) > 5
        assert trace.exit_reason == "passed"
        assert len(trace.iterations) == 1

    def test_loops_on_failure_then_passes(self):
        pipeline, router, gen, _ = _build_pipeline(pass_immediately=False)
        with mock.patch("criticrag.utils.tracer.LOG_TRACES", False), \
             mock.patch("criticrag.agents.router.generate", return_value="reformulated aurora query"):
            answer, trace = pipeline.run(QUERY)
        assert trace.exit_reason == "passed"
        assert len(trace.iterations) == 2
        assert trace.iterations[0].reformulated_query is not None

    def test_trace_has_all_fields(self):
        pipeline, _, _, _ = _build_pipeline(pass_immediately=True)
        with mock.patch("criticrag.utils.tracer.LOG_TRACES", False):
            _, trace = pipeline.run(QUERY)
        it = trace.iterations[0]
        assert it.retrieval.strategy in ("dense", "sparse", "hybrid")
        assert it.critic.mean_score > 0
        assert it.answer_latency_ms >= 0
        assert trace.total_latency_ms >= 0

class TestNaiveRAGPipeline:
    def test_returns_string(self):
        sparse = SparseIndex(CHUNKS)
        dense_mock = mock.MagicMock()
        dense_mock.search.return_value = [(0.9, "aurora", CHUNKS[0][1])]
        hybrid_mock = mock.MagicMock()
        hybrid_mock.search.return_value = [(0.9, "aurora", CHUNKS[0][1])]
        router = RouterAgent(dense_mock, sparse, hybrid_mock)

        gen = mock.MagicMock(spec=GeneratorAgent)
        gen.answer.return_value = "Solar wind causes aurora."

        naive = NaiveRAGPipeline(router, gen)
        result = naive.run(QUERY)
        assert isinstance(result, str)
        assert len(result) > 0
