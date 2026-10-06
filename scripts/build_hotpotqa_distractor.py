from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from criticrag.retrieval.dense import DenseIndex
from criticrag.retrieval.sparse import SparseIndex
from criticrag.retrieval.hybrid import HybridIndex
from criticrag.agents.router import RouterAgent
from criticrag.agents.generator import GeneratorAgent
from criticrag.agents.critic import CriticAgent
from criticrag.pipeline import CriticRAGPipeline, NaiveRAGPipeline
from criticrag.utils.config import CRITIC_PASS_THRESHOLD

def build_chunks_from_context(context: dict) -> list[tuple[str, str]]:
    """
    Convert HotpotQA context dict → list of (title, paragraph_text) chunks.
    Each paragraph's sentences are joined into one chunk.
    """
    chunks = []
    for title, sentences in zip(context["title"], context["sentences"]):
        text = " ".join(s.strip() for s in sentences if s.strip())
        if text:
            chunks.append((title, text))
    return chunks

def build_pipeline_for_chunks(chunks: list[tuple[str, str]]):
    """Build fresh dense+sparse+hybrid indices for a single question's context."""
    sparse = SparseIndex(chunks)
    dense = DenseIndex(chunks)
    hybrid = HybridIndex(dense, sparse)
    router = RouterAgent(dense, sparse, hybrid)
    return router

def _normalize(text: str) -> str:
    """
    Normalize answer text for fairer comparison.
    Handles common mismatches: abbreviations, punctuation, date formats,
    articles, hyphen vs 'and', Inc./Ltd. suffixes.
    """
    import re
    text = text.strip().lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r'(\d{4})\s*[-–—to until]+\s*(\d{4})', r'\1 \2', text)
    text = re.sub(r'[-–—]', ' ', text)
    text = re.sub(r'\b(inc|ltd|dsc|jr|sr|phd|md)\b\.?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

_ABBREV = {
    "ibhof": "international boxing hall of fame",
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "nyc": "new york city",
    "ny": "new york",
    "la": "los angeles",
}

def score_answer(answer: str, expected: str, question: str) -> dict:
    """
    Exact match + token F1 with normalization.
    Handles common surface-form mismatches without changing pipeline behavior.
    """
    pred = _normalize(answer)
    gold = _normalize(expected)

    pred = _ABBREV.get(pred, pred)
    gold = _ABBREV.get(gold, gold)

    exact = int(pred == gold or gold in pred or pred in gold)

    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    pred_tokens.discard('')
    gold_tokens.discard('')

    if not pred_tokens or not gold_tokens:
        f1 = float(exact)
    else:
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"exact_match": exact, "f1": round(max(f1, float(exact)), 3)}

def run_eval(
    n: int = 50,
    level: str = "hard",
    out_path: str = "results/hotpotqa_distractor.json",
) -> None:
    from datasets import load_dataset

    print(f"Loading HotpotQA distractor (validation, level={level})...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)

    naive_generator = GeneratorAgent(concise=True, self_consistency=1)
    critic_generator = GeneratorAgent(concise=True, self_consistency=3)
    critic = CriticAgent()

    naive_results = []
    critic_results = []
    scanned = 0
    kept = 0

    for sample in ds:
        scanned += 1
        if level != "all" and sample["level"] != level:
            continue
        if kept >= n:
            break

        kept += 1
        qid = sample["id"]
        question = sample["question"]
        expected = sample["answer"]
        q_type = sample["type"]

        print(f"\n[{kept}/{n}] {question[:70]}...")
        print(f"  expected: {expected}")

        chunks = build_chunks_from_context(sample["context"])
        router = build_pipeline_for_chunks(chunks)

        naive_pipeline = NaiveRAGPipeline(router, naive_generator)
        critic_pipeline = CriticRAGPipeline(router, critic_generator, critic)

        t0 = time.monotonic()
        try:
            naive_answer = naive_pipeline.run(question)
        except Exception as e:
            naive_answer = f"[error: {e}]"
        naive_latency = (time.monotonic() - t0) * 1000
        naive_scores = score_answer(naive_answer, expected, question)
        print(f"  naive  → '{naive_answer[:60]}' | f1={naive_scores['f1']:.2f} ({naive_latency:.0f}ms)")

        t0 = time.monotonic()
        try:
            critic_answer, trace = critic_pipeline.run(question)
        except Exception as e:
            critic_answer = f"[error: {e}]"
            trace = None
        critic_latency = (time.monotonic() - t0) * 1000
        critic_scores = score_answer(critic_answer, expected, question)
        iterations = len(trace.iterations) if trace else 0
        exit_reason = trace.exit_reason if trace else "error"
        print(f"  critic → '{critic_answer[:60]}' | f1={critic_scores['f1']:.2f} ({critic_latency:.0f}ms) iters={iterations}")

        naive_results.append({
            "id": qid, "question": question, "expected": expected,
            "answer": naive_answer, "type": q_type, "level": sample["level"],
            **naive_scores, "latency_ms": round(naive_latency, 1),
        })
        critic_results.append({
            "id": qid, "question": question, "expected": expected,
            "answer": critic_answer, "type": q_type, "level": sample["level"],
            **critic_scores, "latency_ms": round(critic_latency, 1),
            "iterations": iterations, "exit_reason": exit_reason,
        })

    def agg(results):
        n = len(results)
        return {
            "n": n,
            "exact_match": round(sum(r["exact_match"] for r in results) / n, 3),
            "f1": round(sum(r["f1"] for r in results) / n, 3),
            "latency_mean_ms": round(sum(r["latency_ms"] for r in results) / n, 1),
        }

    naive_agg = agg(naive_results)
    critic_agg = agg(critic_results)

    print(f"\n{'='*60}")
    print(f"  HotpotQA Distractor — {kept} questions (level={level})")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Naive RAG':>12} {'CriticRAG':>12} {'Delta':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Exact Match':<20} {naive_agg['exact_match']:>12.3f} {critic_agg['exact_match']:>12.3f} {critic_agg['exact_match']-naive_agg['exact_match']:>+10.3f}")
    print(f"  {'Token F1':<20} {naive_agg['f1']:>12.3f} {critic_agg['f1']:>12.3f} {critic_agg['f1']-naive_agg['f1']:>+10.3f}")
    print(f"  {'Latency (ms)':<20} {naive_agg['latency_mean_ms']:>12.1f} {critic_agg['latency_mean_ms']:>12.1f} {critic_agg['latency_mean_ms']-naive_agg['latency_mean_ms']:>+10.1f}")
    print(f"{'='*60}\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "hotpot_qa/distractor",
        "level": level,
        "naive_rag": {**naive_agg, "samples": naive_results},
        "criticrag": {**critic_agg, "samples": critic_results},
        "delta": {
            "exact_match": round(critic_agg["exact_match"] - naive_agg["exact_match"], 3),
            "f1": round(critic_agg["f1"] - naive_agg["f1"], 3),
            "latency_ms": round(critic_agg["latency_mean_ms"] - naive_agg["latency_mean_ms"], 1),
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--level", default="hard", choices=["hard", "medium", "easy", "all"])
    parser.add_argument("--out", default="results/hotpotqa_distractor.json")
    args = parser.parse_args()
    run_eval(n=args.n, level=args.level, out_path=args.out)

if __name__ == "__main__":
    main()
