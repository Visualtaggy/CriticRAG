
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from criticrag.utils.corpus import build_indices
from criticrag.agents.generator import GeneratorAgent
from criticrag.agents.critic import CriticAgent
from criticrag.pipeline import CriticRAGPipeline

def main() -> None:
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    query = " ".join(args).strip()
    if not query:
        print('Usage: python -m criticrag [--debug] "your question here"')
        sys.exit(1)

    print("Loading corpus and building indices...")
    _, _, _, _, router = build_indices()

    pipeline = CriticRAGPipeline(router, GeneratorAgent(), CriticAgent())

    print(f"\nQ: {query}\n")
    answer, trace = pipeline.run(query)

    if debug:
        print("── Retrieved chunks (iteration 0) ──────────────────────")
        for score, src, text in trace.iterations[0].retrieval.chunks:
            print(f"  {score:.3f} [{src}] {text[:80]}")
        print()

    print(f"A: {answer}\n")
    print(f"── Trace summary ───────────────────────────────────────")
    print(f"  Iterations   : {len(trace.iterations)}")
    print(f"  Exit reason  : {trace.exit_reason}")
    print(f"  Total time   : {trace.total_latency_ms:.0f} ms")
    for it in trace.iterations:
        print(f"  [{it.index}] strategy={it.retrieval.strategy} "
              f"score={it.critic.mean_score:.1f} "
              f"passed={it.critic.passed} "
              f"failure={it.critic.failure_type}")
    if trace.exit_reason == "passed":
        winning = trace.iterations[-1]
        print(f"\n  Final scores: {winning.critic.scores}")
    print()

if __name__ == "__main__":
    main()
