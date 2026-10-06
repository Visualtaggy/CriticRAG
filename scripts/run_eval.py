from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from criticrag.utils.corpus import build_indices
from criticrag.agents.generator import GeneratorAgent
from criticrag.agents.critic import CriticAgent
from criticrag.pipeline import CriticRAGPipeline, NaiveRAGPipeline
from criticrag.eval.harness import EvalHarness

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_set.jsonl")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--out", default="results", help="Directory for report JSON files")
    args = parser.parse_args()

    print("Loading corpus and building indices...")
    _, _, _, _, router = build_indices(args.corpus)

    generator = GeneratorAgent()
    critic = CriticAgent()

    critic_pipeline = CriticRAGPipeline(router, generator, critic)
    naive_pipeline = NaiveRAGPipeline(router, generator)

    harness = EvalHarness(eval_set_path=args.eval_set)

    print("\nRunning Naive RAG baseline...")
    naive_report = harness.run(naive_pipeline, pipeline_name="Naive RAG")
    naive_report.print_summary()

    print("Running CriticRAG...")
    critic_report = harness.run(critic_pipeline, pipeline_name="CriticRAG")
    critic_report.print_summary()

    delta_score = critic_report.mean_score - naive_report.mean_score
    delta_pass = critic_report.pass_rate - naive_report.pass_rate
    print("── Delta (CriticRAG − Naive) ──────────────────────────────")
    print(f"  Mean score   : {delta_score:+.2f}")
    print(f"  Pass rate    : {delta_pass:+.1%}")
    print(f"  Latency cost : +{critic_report.mean_latency_ms - naive_report.mean_latency_ms:.0f} ms / query")
    print()

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        naive_report.save(os.path.join(args.out, "naive_report.json"))
        critic_report.save(os.path.join(args.out, "critic_report.json"))
        print(f"Reports saved to {args.out}/")

if __name__ == "__main__":
    main()
