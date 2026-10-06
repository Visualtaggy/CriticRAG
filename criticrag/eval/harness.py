from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..agents.critic import CriticAgent
from ..utils.ollama import judge
from ..utils.config import EVAL_SET_PATH, CRITIC_PASS_THRESHOLD

_RUBRIC_JUDGE_PROMPT = """\
You are evaluating a RAG system answer against a rubric.

Question: {query}
Expected answer / rubric: {rubric}
Actual answer: {answer}

Score the answer on each axis from 0 to 10.
Return ONLY valid JSON, no prose:
{{
  "faithfulness": <int>,
  "relevance": <int>,
  "completeness": <int>,
  "reasoning": "<one sentence>"
}}"""

@runtime_checkable
class RunnablePipeline(Protocol):
    def run(self, query: str) -> tuple[str, object]: ...

@runtime_checkable
class SimplePipeline(Protocol):
    def run(self, query: str) -> str: ...

@dataclass
class EvalSample:
    id: str
    query: str
    answer: str
    scores: dict[str, float]
    mean_score: float
    passed: bool
    latency_ms: float
    reasoning: str = ""

@dataclass
class EvalReport:
    pipeline_name: str
    n_samples: int
    pass_rate: float
    mean_faithfulness: float
    mean_relevance: float
    mean_completeness: float
    mean_score: float
    mean_latency_ms: float
    samples: list[EvalSample] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.pipeline_name}")
        print(f"{'='*60}")
        print(f"  Samples        : {self.n_samples}")
        print(f"  Pass rate      : {self.pass_rate:.1%}")
        print(f"  Mean score     : {self.mean_score:.2f} / 10")
        print(f"  Faithfulness   : {self.mean_faithfulness:.2f}")
        print(f"  Relevance      : {self.mean_relevance:.2f}")
        print(f"  Completeness   : {self.mean_completeness:.2f}")
        print(f"  Latency (mean) : {self.mean_latency_ms:.0f} ms")
        print(f"{'='*60}\n")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

class EvalHarness:
    def __init__(self, eval_set_path: str = EVAL_SET_PATH):
        self.eval_set = self._load(eval_set_path)
        self._critic_call = judge

    def _load(self, path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Eval set not found: {path}")
        samples = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _score_answer(self, query: str, rubric: str, answer: str) -> tuple[dict, str]:
        prompt = _RUBRIC_JUDGE_PROMPT.format(query=query, rubric=rubric, answer=answer)
        raw = self._critic_call(prompt)
        try:
            import re
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            data = json.loads(m.group()) if m else {}
            scores = {
                "faithfulness": float(data.get("faithfulness", 0)),
                "relevance": float(data.get("relevance", 0)),
                "completeness": float(data.get("completeness", 0)),
            }
            reasoning = data.get("reasoning", "")
        except Exception:
            scores = {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0}
            reasoning = f"Parse error: {raw[:100]}"
        return scores, reasoning

    def run(self, pipeline, pipeline_name: str = "pipeline") -> EvalReport:
        """
        pipeline: anything with a .run(query) -> (answer, trace) or .run(query) -> str
        """
        results: list[EvalSample] = []

        for item in self.eval_set:
            qid = item.get("id", "?")
            query = item["query"]
            rubric = item.get("rubric", item.get("expected", ""))

            t0 = time.monotonic()
            try:
                out = pipeline.run(query)
                answer = out[0] if isinstance(out, tuple) else out
            except Exception as e:
                answer = f"[pipeline error: {e}]"
            latency_ms = (time.monotonic() - t0) * 1000

            scores, reasoning = self._score_answer(query, rubric, answer)
            mean = sum(scores.values()) / 3

            results.append(EvalSample(
                id=qid,
                query=query,
                answer=answer,
                scores=scores,
                mean_score=round(mean, 2),
                passed=mean >= CRITIC_PASS_THRESHOLD,
                latency_ms=round(latency_ms, 1),
                reasoning=reasoning,
            ))
            print(f"  [{qid}] score={mean:.1f} latency={latency_ms:.0f}ms")

        n = len(results)
        axes = ["faithfulness", "relevance", "completeness"]

        return EvalReport(
            pipeline_name=pipeline_name,
            n_samples=n,
            pass_rate=sum(r.passed for r in results) / n if n else 0,
            mean_faithfulness=sum(r.scores["faithfulness"] for r in results) / n if n else 0,
            mean_relevance=sum(r.scores["relevance"] for r in results) / n if n else 0,
            mean_completeness=sum(r.scores["completeness"] for r in results) / n if n else 0,
            mean_score=sum(r.mean_score for r in results) / n if n else 0,
            mean_latency_ms=sum(r.latency_ms for r in results) / n if n else 0,
            samples=results,
        )
