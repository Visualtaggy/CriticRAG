from __future__ import annotations

from ..utils.ollama import generate
from ..utils.config import GEN_MODEL

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer using ONLY the provided context. "
    "Cite sources inline as [1], [2], etc. "
    "If the context does not contain enough information to answer, "
    "say exactly: 'The provided context does not contain enough information.'"
)

_SYSTEM_CONCISE = (
    "You are a precise question-answering assistant. "
    "Answer using ONLY the provided context. "
    "OUTPUT RULES: maximum 5 words. A name, date, number, or yes/no only. "
    "NO sentences. NO explanations. NO punctuation beyond what is in the answer itself. "
    "If the context does not contain the answer, output exactly: Unknown"
)

_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {query}

Answer:"""

class GeneratorAgent:
    def __init__(self, model: str = GEN_MODEL, concise: bool = False,
                 self_consistency: int = 1):
        self.model = model
        self.concise = concise
        self.self_consistency = self_consistency

    def _build_context(self, hits: list[tuple[float, str, str]]) -> str:
        lines = []
        for i, (_, source, text) in enumerate(hits, start=1):
            lines.append(f"[{i}] ({source}) {text}")
        return "\n".join(lines)

    def _majority_vote(self, answers: list[str]) -> str:
        """Return the most common answer. Ties broken by first occurrence."""
        from collections import Counter
        normalized = [a.strip().lower() for a in answers]
        counts = Counter(normalized)
        winner = counts.most_common(1)[0][0]
        for a in answers:
            if a.strip().lower() == winner:
                return a
        return answers[0]

    def answer(self, query: str, hits: list[tuple[float, str, str]]) -> str:
        context = self._build_context(hits)
        prompt = _PROMPT_TEMPLATE.format(context=context, query=query)
        system = _SYSTEM_CONCISE if self.concise else _SYSTEM

        if self.self_consistency <= 1:
            return generate(prompt, model=self.model, system=system)

        answers = [
            generate(prompt, model=self.model, system=system)
            for _ in range(self.self_consistency)
        ]
        return self._majority_vote(answers)
