
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..utils.config import TOP_K

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class SparseIndex:
    """
    BM25 index over a list of (source, text) chunks.
    """

    def __init__(self, chunks: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build(chunks)

    def _build(self, chunks: list[tuple[str, str]]) -> None:
        tokenized = [_tokenize(text) for _, text in chunks]
        N = len(tokenized)
        lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl = float(lengths.mean()) if N > 0 else 1.0

        vocab = sorted({w for doc in tokenized for w in doc})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)

        df = np.zeros(V, dtype=np.float32)
        for doc in tokenized:
            for w in set(doc):
                if w in self.vocab:
                    df[self.vocab[w]] += 1

        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        tf = np.zeros((N, V), dtype=np.float32)
        for i, doc in enumerate(tokenized):
            for w, cnt in Counter(doc).items():
                if w in self.vocab:
                    tf[i, self.vocab[w]] = cnt

        k1, b = self.k1, self.b
        dl_ratio = lengths[:, None] / self.avgdl
        self.matrix = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl_ratio))

    def _encode_query(self, query: str) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for w in _tokenize(query):
            if w in self.vocab:
                v[self.vocab[w]] = 1.0
        return v

    def search(self, query: str, k: int = TOP_K) -> list[tuple[float, str, str]]:
        """Returns [(score, source, text), ...] sorted desc."""
        qvec = self._encode_query(query) * self.idf
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.matrix.astype(np.float64) @ qvec.astype(np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i][0], self.chunks[i][1]) for i in idx]
