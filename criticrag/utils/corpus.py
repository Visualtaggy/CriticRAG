"""
Load corpus.jsonl into a list of (source, text) tuples.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import CORPUS_PATH

def load_corpus(path: str = CORPUS_PATH) -> list[tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run scripts/build_corpus.py first."
        )
    chunks = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                chunks.append((obj["source"], obj["text"]))
    return chunks

def build_indices(corpus_path: str = CORPUS_PATH):
    """
    Convenience: load corpus and build all three indices.
    Returns (chunks, dense, sparse, hybrid, router).
    """
    from ..retrieval.dense import DenseIndex
    from ..retrieval.sparse import SparseIndex
    from ..retrieval.hybrid import HybridIndex
    from ..agents.router import RouterAgent

    chunks = load_corpus(corpus_path)
    print(f"Loaded {len(chunks)} chunks.")

    print("Building sparse index...")
    sparse = SparseIndex(chunks)

    print("Building dense index (calls Ollama — may take a minute)...")
    dense = DenseIndex(chunks)

    hybrid = HybridIndex(dense, sparse)
    router = RouterAgent(dense, sparse, hybrid)

    return chunks, dense, sparse, hybrid, router
