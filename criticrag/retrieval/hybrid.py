
from __future__ import annotations

from .dense import DenseIndex
from .sparse import SparseIndex
from ..utils.config import TOP_K

RRF_K = 60

class HybridIndex:


    def __init__(self, dense: DenseIndex, sparse: SparseIndex):
        self.dense = dense
        self.sparse = sparse

    def search(self, query: str, k: int = TOP_K) -> list[tuple[float, str, str]]:
        """Returns [(rrf_score, source, text), ...] sorted desc."""
        fetch = max(k * 3, 20)
        dense_hits = self.dense.search(query, k=fetch)
        sparse_hits = self.sparse.search(query, k=fetch)

        def key(src: str, text: str) -> str:
            return f"{src}::{text[:40]}"

        rrf: dict[str, float] = {}
        chunk_map: dict[str, tuple[str, str]] = {}

        for rank, (_, src, text) in enumerate(dense_hits, start=1):
            k_ = key(src, text)
            rrf[k_] = rrf.get(k_, 0.0) + 1.0 / (RRF_K + rank)
            chunk_map[k_] = (src, text)

        for rank, (_, src, text) in enumerate(sparse_hits, start=1):
            k_ = key(src, text)
            rrf[k_] = rrf.get(k_, 0.0) + 1.0 / (RRF_K + rank)
            chunk_map[k_] = (src, text)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(score, *chunk_map[ck]) for ck, score in ranked]
