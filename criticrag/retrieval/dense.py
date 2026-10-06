
from __future__ import annotations

import numpy as np

from ..utils.ollama import embed
from ..utils.config import TOP_K, EMBED_BATCH_SIZE

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / n, v)

class DenseIndex:

    def __init__(self, chunks: list[tuple[str, str]]):
        self.chunks = chunks
        self.matrix = self._build(chunks)

    def _build(self, chunks: list[tuple[str, str]]) -> np.ndarray:
        vecs = []
        texts = [f"search_document: {text}" for _, text in chunks]
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            vecs.append(embed(batch))
        raw = np.vstack(vecs)
        normed = _normalize(raw)
        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        return normed

    def search(self, query: str, k: int = TOP_K) -> list[tuple[float, str, str]]:
        """Returns [(score, source, text), ...] sorted desc."""
        qvec = _normalize(embed([f"search_query: {query}"]))[0]
        qvec = np.nan_to_num(qvec, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.matrix.astype(np.float64) @ qvec.astype(np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i][0], self.chunks[i][1]) for i in idx]
