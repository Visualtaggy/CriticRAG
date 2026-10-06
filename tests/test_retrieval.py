"""
Tests for sparse and hybrid retrieval.
Dense tests are skipped unless Ollama is reachable (CI-friendly).
"""
from __future__ import annotations

import pytest
import numpy as np

from criticrag.retrieval.sparse import SparseIndex
from criticrag.retrieval.hybrid import HybridIndex
from criticrag.agents.router import RouterAgent

CHUNKS = [
    ("aurora",      "Solar wind particles interact with Earth's magnetic field causing auroras."),
    ("photosyn",    "Plants convert sunlight and carbon dioxide into glucose via photosynthesis."),
    ("bm25",        "BM25 is a term-frequency ranking function used in information retrieval."),
    ("transformer", "Transformers use self-attention to model long-range dependencies in sequences."),
    ("crispr",      "CRISPR-Cas9 uses guide RNA to direct a protein to cut specific DNA sequences."),
]

class TestSparseIndex:
    def setup_method(self):
        self.idx = SparseIndex(CHUNKS)

    def test_returns_k_results(self):
        hits = self.idx.search("solar wind magnetic field", k=3)
        assert len(hits) == 3

    def test_top_hit_is_relevant(self):
        hits = self.idx.search("BM25 term frequency ranking")
        top_src = hits[0][1]
        assert top_src == "bm25"

    def test_scores_are_nonneg(self):
        for score, _, _ in self.idx.search("photosynthesis glucose"):
            assert score >= 0

    def test_aurora_query(self):
        hits = self.idx.search("aurora solar wind")
        sources = [src for _, src, _ in hits]
        assert "aurora" in sources[:2]

    def test_empty_query_returns_results(self):
        hits = self.idx.search("")
        assert isinstance(hits, list)

class TestRouterHeuristics:
    def setup_method(self):
        sparse = SparseIndex(CHUNKS)
        from criticrag.retrieval.dense import DenseIndex
        import unittest.mock as mock

        dense_mock = mock.MagicMock(spec=DenseIndex)
        dense_mock.search.return_value = []
        dense_mock.chunks = CHUNKS
        dense_mock.matrix = np.zeros((len(CHUNKS), 4))

        hybrid_mock = mock.MagicMock(spec=HybridIndex)
        hybrid_mock.search.return_value = []

        self.router = RouterAgent(dense_mock, sparse, hybrid_mock)

    def test_short_query_goes_sparse(self):
        assert self.router.select_strategy("CRISPR Cas9") == "sparse"

    def test_question_goes_dense(self):
        assert self.router.select_strategy("How does photosynthesis work?") == "dense"

    def test_long_keyword_query_goes_sparse(self):
        strategy = self.router.select_strategy("BM25 TF-IDF ranking normalization saturation")
        assert strategy in ("sparse", "hybrid")

    def test_natural_language_goes_dense(self):
        strategy = self.router.select_strategy("Why do transformers perform well on long sequences?")
        assert strategy == "dense"
