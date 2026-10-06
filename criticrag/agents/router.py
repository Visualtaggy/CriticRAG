from __future__ import annotations

import re

from ..retrieval.dense import DenseIndex
from ..retrieval.sparse import SparseIndex
from ..retrieval.hybrid import HybridIndex
from ..agents.tools import search_wikipedia, calculate
from ..utils.ollama import generate
from ..utils.config import (
    SHORT_QUERY_THRESHOLD,
    KEYWORD_DENSITY_THRESHOLD,
    TOP_K,
)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "it", "its", "this", "that", "these", "those", "and", "or", "but",
    "if", "not", "no", "so", "than", "then", "there", "what", "which",
    "who", "whom", "how", "when", "where", "why",
}

_QUESTION_WORDS = {"how", "why", "explain", "describe", "what", "compare",
                   "summarise", "summarize", "which", "who", "where", "when"}

_MATH_PATTERN = re.compile(r"[\d\+\-\*\/\(\)\^\.]+\s*[\+\-\*\/\^]\s*[\d\+\-\*\/\(\)\^\.]+")

_REFORMULATE_PROMPT = """\
The following query failed to retrieve a useful answer from a knowledge base.
Failure type: {failure_type}
Failure reason: {failure_reason}
Original query: {query}

Rewrite the query to be more likely to retrieve relevant information.
Return ONLY the rewritten query, nothing else."""

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _is_math(query: str) -> bool:
    return bool(_MATH_PATTERN.search(query))

def _proper_noun_density(query: str) -> float:
    """
    Fraction of non-first words that start with a capital letter.
    Skips the first word (always capitalised) and stopwords.
    """
    words = query.split()
    if len(words) < 2:
        return 0.0
    rest = words[1:]
    proper = sum(1 for w in rest if w and w[0].isupper() and w.lower() not in _STOPWORDS)
    return proper / len(rest)

class RouterAgent:
    """
    Tool-calling router: selects retrieval strategy or external tool per query.

    Usage:
        router = RouterAgent(dense_index, sparse_index, hybrid_index)
        strategy, hits = router.retrieve(query)
    """

    def __init__(self, dense: DenseIndex, sparse: SparseIndex, hybrid: HybridIndex,
                 enable_tools: bool = True):
        self.dense = dense
        self.sparse = sparse
        self.hybrid = hybrid
        self.enable_tools = enable_tools

    def select_strategy(self, query: str, failure_type: str | None = None) -> str:
        if self.enable_tools and _is_math(query):
            return "calculate"

        tokens = _tokenize(query)
        if not tokens:
            return "hybrid"

        n = len(tokens)
        content_words = [t for t in tokens if t not in _STOPWORDS]
        keyword_density = len(content_words) / n
        question_words_present = bool(_QUESTION_WORDS & set(tokens))

        has_proper_nouns = _proper_noun_density(query) > 0.25
        if (self.enable_tools
                and failure_type is not None
                and has_proper_nouns):
            return "search_wikipedia"

        if question_words_present:
            return "dense"

        if n <= SHORT_QUERY_THRESHOLD:
            return "sparse"

        if keyword_density > KEYWORD_DENSITY_THRESHOLD:
            return "sparse"

        return "hybrid"

    def retrieve(
        self,
        query: str,
        strategy: str | None = None,
        k: int = TOP_K,
        failure_type: str | None = None,
    ) -> tuple[str, list[tuple[float, str, str]]]:
        """
        Returns (strategy_used, hits).
        hits: [(score, source, text), ...]
        """
        if strategy is None:
            strategy = self.select_strategy(query, failure_type=failure_type)

        if strategy == "calculate":
            return strategy, calculate(query)
        if strategy == "search_wikipedia":
            return strategy, search_wikipedia(query)
        if strategy == "dense":
            return strategy, self.dense.search(query, k=k)
        if strategy == "sparse":
            return strategy, self.sparse.search(query, k=k)
        return "hybrid", self.hybrid.search(query, k=k)

    def reformulate(self, query: str, failure_type: str, failure_reason: str) -> str:
        prompt = _REFORMULATE_PROMPT.format(
            failure_type=failure_type,
            failure_reason=failure_reason,
            query=query,
        )
        return generate(prompt).strip().strip('"').strip("'")
