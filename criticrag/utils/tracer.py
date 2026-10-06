"""
Structured execution tracer.

Every pipeline run produces a trace dict that records:
  - the original query
  - each iteration: strategy chosen, chunks retrieved, answer, critic scores,
    failure type (if any), reformulated query (if any)
  - final answer and exit reason
  - wall-clock latency per step

Traces are written to TRACES_DIR as JSON files (one per query).
They are also returned inline so callers can inspect them programmatically.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .config import TRACES_DIR, LOG_TRACES, TRACE_INDENT

@dataclass
class RetrievalStep:
    strategy: str                          # "dense" | "sparse" | "hybrid"
    query_used: str
    chunks: list[tuple[float, str, str]]
    latency_ms: float

@dataclass
class CriticStep:
    scores: dict[str, float]
    mean_score: float
    passed: bool
    failure_type: str | None
    failure_reason: str | None
    latency_ms: float

@dataclass
class Iteration:
    index: int
    retrieval: RetrievalStep
    answer: str
    answer_latency_ms: float
    critic: CriticStep
    reformulated_query: str | None = None

@dataclass
class Trace:
    query: str
    iterations: list[Iteration] = field(default_factory=list)
    final_answer: str = ""
    exit_reason: str = ""          # "passed" | "max_iterations" | "error"
    total_latency_ms: float = 0.0
    _start: float = field(default_factory=time.monotonic, repr=False)

    def add_iteration(self, it: Iteration) -> None:
        self.iterations.append(it)

    def finish(self, answer: str, exit_reason: str) -> None:
        self.final_answer = answer
        self.exit_reason = exit_reason
        self.total_latency_ms = (time.monotonic() - self._start) * 1000

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_start", None)
        return d

    def save(self, query_id: str | None = None) -> str | None:
        if not LOG_TRACES:
            return None
        os.makedirs(TRACES_DIR, exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in self.query[:40])
        fname = f"{query_id or safe}.json"
        path = os.path.join(TRACES_DIR, fname)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=TRACE_INDENT)
        return path

class Timer:
    def __enter__(self):
        self._t = time.monotonic()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.monotonic() - self._t) * 1000
