from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from criticrag.utils.corpus import build_indices
from criticrag.agents.generator import GeneratorAgent
from criticrag.agents.critic import CriticAgent
from criticrag.pipeline import CriticRAGPipeline

DEMO_QUERIES = [
    {
        "query": "What is dark matter?",
        "expected_tool": "dense",
        "note": "corpus hit — dense retrieval",
    },
    {
        "query": "aurora borealis solar wind magnetic field",
        "expected_tool": "sparse",
        "note": "corpus hit — sparse retrieval",
    },
    {
        "query": "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger?",
        "expected_tool": "search_wikipedia",
        "note": "corpus miss → Wikipedia tool",
    },
    {
        "query": "What government position was held by Shirley Temple?",
        "expected_tool": "search_wikipedia",
        "note": "corpus miss → Wikipedia tool",
    },
    {
        "query": "384 * 12 + 7",
        "expected_tool": "calculate",
        "note": "math expression → calculator tool",
    },
    {
        "query": "2 ^ 10",
        "expected_tool": "calculate",
        "note": "math expression → calculator tool",
    },
    {
        "query": "How did the Cold War affect global politics and alliances?",
        "expected_tool": "dense",
        "note": "corpus hit — dense retrieval",
    },
]

SEPARATOR = "─" * 65

def run_demo(corpus_path: str) -> None:
    print("Loading corpus and building indices...")
    _, _, _, _, router = build_indices(corpus_path)
    generator = GeneratorAgent(concise=False)
    critic = CriticAgent()
    pipeline = CriticRAGPipeline(router, generator, critic)

    print(f"\n{'='*65}")
    print("  CriticRAG Tool-Calling Demo")
    print(f"{'='*65}\n")

    for i, item in enumerate(DEMO_QUERIES, 1):
        query = item["query"]
        note = item["note"]
        expected = item["expected_tool"]

        print(f"[{i}/{len(DEMO_QUERIES)}] {note}")
        print(f"  Query    : {query}")

        answer, trace = pipeline.run(query)

        for it in trace.iterations:
            marker = "✓" if it.critic.passed else "✗"
            tool_marker = " ← TOOL" if it.retrieval.strategy in ("search_wikipedia", "calculate") else ""
            failure_str = f" failure={it.critic.failure_type}" if not it.critic.passed else ""
            print(f"  Iter {it.index}   : strategy={it.retrieval.strategy}{tool_marker} "
                  f"score={it.critic.mean_score:.1f}{failure_str} {marker}")
            if it.retrieval.strategy == "search_wikipedia":
                print(f"  Wiki hit : {it.retrieval.chunks[0][1] if it.retrieval.chunks else 'none'}")
            elif it.retrieval.strategy == "calculate":
                print(f"  Calc     : {it.retrieval.chunks[0][2] if it.retrieval.chunks else 'none'}")
            if it.reformulated_query:
                print(f"  Rewrite  : {it.reformulated_query}")

        all_strategies = [it.retrieval.strategy for it in trace.iterations]
        actual_tool = next(
            (s for s in all_strategies if s in ("search_wikipedia", "calculate")),
            all_strategies[0] if all_strategies else "none"
        )
        match = "✓" if actual_tool == expected else f"✗ (expected {expected})"
        print(f"  Tool     : {actual_tool} {match}")
        print(f"  Answer   : {answer[:120]}")
        print(f"  Exit     : {trace.exit_reason} ({len(trace.iterations)} iter, {trace.total_latency_ms:.0f}ms)")
        print(SEPARATOR)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    args = parser.parse_args()
    run_demo(args.corpus)

if __name__ == "__main__":
    main()
