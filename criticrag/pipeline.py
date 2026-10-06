from __future__ import annotations

from .agents.router import RouterAgent
from .agents.generator import GeneratorAgent
from .agents.critic import CriticAgent
from .utils.tracer import Trace, Iteration, RetrievalStep, CriticStep, Timer
from .utils.config import MAX_LOOP_ITERATIONS, TOP_K

class CriticRAGPipeline:
    def __init__(
        self,
        router: RouterAgent,
        generator: GeneratorAgent | None = None,
        critic: CriticAgent | None = None,
    ):
        self.router = router
        self.generator = generator or GeneratorAgent()
        self.critic = critic or CriticAgent()

    def run(self, query: str, k: int = TOP_K) -> tuple[str, Trace]:
        """
        Run the full pipeline.

        Returns:
            answer: str
            trace:  Trace — full structured execution log
        """
        trace = Trace(query=query)
        current_query = query
        strategy_override: str | None = None
        best_answer: str = ""
        best_score: float | None = None
        last_failure_type: str | None = None

        for iteration_idx in range(MAX_LOOP_ITERATIONS):

            with Timer() as t_ret:
                strategy, hits = self.router.retrieve(
                    current_query, strategy=strategy_override, k=k,
                    failure_type=last_failure_type,
                )
            ret_step = RetrievalStep(
                strategy=strategy,
                query_used=current_query,
                chunks=hits,
                latency_ms=t_ret.elapsed_ms,
            )

            with Timer() as t_gen:
                answer = self.generator.answer(current_query, hits)

            with Timer() as t_crit:
                result = self.critic.evaluate(current_query, hits, answer)

            crit_step = CriticStep(
                scores=result.scores,
                mean_score=result.mean_score,
                passed=result.passed,
                failure_type=result.failure_type,
                failure_reason=result.failure_reason,
                latency_ms=t_crit.elapsed_ms,
            )

            iteration = Iteration(
                index=iteration_idx,
                retrieval=ret_step,
                answer=answer,
                answer_latency_ms=t_gen.elapsed_ms,
                critic=crit_step,
            )

            if best_score is None or result.mean_score > best_score:
                best_score = result.mean_score
                best_answer = answer

            if result.passed:
                trace.add_iteration(iteration)
                trace.finish(best_answer, exit_reason="passed")
                trace.save()
                return best_answer, trace

            if iteration_idx == MAX_LOOP_ITERATIONS - 1:
                trace.add_iteration(iteration)
                trace.finish(best_answer, exit_reason="max_iterations")
                trace.save()
                return best_answer, trace

            new_query = self.router.reformulate(
                current_query,
                failure_type=result.failure_type or "incomplete",
                failure_reason=result.failure_reason or "",
            )
            iteration.reformulated_query = new_query
            trace.add_iteration(iteration)

            last_failure_type = result.failure_type

            if result.failure_type == "no_docs":
                strategy_override = None
            elif result.failure_type == "hallucination":
                strategy_override = "dense"
            else:
                strategy_override = None

            current_query = new_query

        trace.finish(answer, exit_reason="max_iterations")
        trace.save()
        return answer, trace

class NaiveRAGPipeline:
    """
    Single retrieve → generate, no critique, no loop.
    Used as the baseline in eval comparisons.
    """

    def __init__(self, router: RouterAgent, generator: GeneratorAgent | None = None):
        self.router = router
        self.generator = generator or GeneratorAgent()

    def run(self, query: str, k: int = TOP_K) -> str:
        _, hits = self.router.retrieve(query)
        return self.generator.answer(query, hits)
