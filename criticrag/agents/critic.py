from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..utils.ollama import judge
from ..utils.config import CRITIC_PASS_THRESHOLD, CRITIC_AXES, FAILURE_TYPES

_SCORE_PROMPT = """\
You are an impartial judge evaluating a RAG system answer.

Question: {query}

Context:
{context}

Answer:
{answer}

Before scoring, reason through these steps internally:
1. List every factual claim in the answer.
2. For each claim, find the supporting sentence in the context or mark it [UNSUPPORTED].
3. Check if the answer addresses what was asked.
4. Check if anything required by the question is missing.

Then output ONLY this JSON and nothing else — no prose before or after:
{{
  "faithfulness": <int 0-10>,
  "relevance": <int 0-10>,
  "completeness": <int 0-10>,
  "reasoning": "<one sentence on the key weakness, or 'none' if all good>"
}}"""

_FAILURE_PROMPT = """\
A RAG system answer failed quality review.

Question: {query}
Context provided: {context}
Answer: {answer}
Critic scores: {scores}

Classify the primary failure reason from these options:
- no_docs: the retrieved context contained no relevant information
- hallucination: the answer made claims not supported by the context
- incomplete: the answer only partially addressed the question

Return ONLY valid JSON:
{{
  "failure_type": "<no_docs|hallucination|incomplete>",
  "failure_reason": "<one sentence describing the specific problem>"
}}"""

@dataclass
class CriticResult:
    scores: dict[str, float]
    mean_score: float
    passed: bool
    failure_type: str | None
    failure_reason: str | None
    reasoning: str

def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in:\n{text}")
    return json.loads(m.group())

def _build_context_str(hits: list[tuple[float, str, str]]) -> str:
    return "\n".join(f"[{i}] ({src}) {txt}" for i, (_, src, txt) in enumerate(hits, 1))

class CriticAgent:
    def evaluate(
        self,
        query: str,
        hits: list[tuple[float, str, str]],
        answer: str,
    ) -> CriticResult:
        context_str = _build_context_str(hits)

        score_prompt = _SCORE_PROMPT.format(
            query=query, context=context_str, answer=answer
        )
        raw_scores = judge(score_prompt)

        try:
            score_data = _extract_json(raw_scores)
            scores = {ax: float(score_data.get(ax, 0)) for ax in CRITIC_AXES}
            reasoning = score_data.get("reasoning", "")
        except (ValueError, KeyError, json.JSONDecodeError):
            scores = {ax: 0.0 for ax in CRITIC_AXES}
            reasoning = f"Failed to parse judge response: {raw_scores[:200]}"

        mean_score = sum(scores.values()) / len(scores)

        answer_tokens = set(answer.lower().split())
        context_text = " ".join(text for _, _, text in hits).lower()
        answer_in_context = any(
            tok in context_text for tok in answer_tokens
            if len(tok) > 3
        )
        short_answer = len(answer.split()) <= 6
        if (short_answer
                and answer_in_context
                and mean_score >= CRITIC_PASS_THRESHOLD - 1.5):
            passed = True
        else:
            passed = mean_score >= CRITIC_PASS_THRESHOLD

        failure_type: str | None = None
        failure_reason: str | None = None

        if not passed:
            fail_prompt = _FAILURE_PROMPT.format(
                query=query,
                context=context_str,
                answer=answer,
                scores=scores,
            )
            raw_fail = judge(fail_prompt)
            try:
                fail_data = _extract_json(raw_fail)
                ft = fail_data.get("failure_type", "incomplete")
                failure_type = ft if ft in FAILURE_TYPES else "incomplete"
                failure_reason = fail_data.get("failure_reason", "")
            except (ValueError, json.JSONDecodeError):
                failure_type = "incomplete"
                failure_reason = "Could not parse failure classification."

        return CriticResult(
            scores=scores,
            mean_score=round(mean_score, 2),
            passed=passed,
            failure_type=failure_type,
            failure_reason=failure_reason,
            reasoning=reasoning,
        )
