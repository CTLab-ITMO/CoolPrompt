"""HyPEROptimizer: HyPE with iterative refinement via recommendations."""

import random
from typing import Any, List, Optional, Sequence, Tuple

from tqdm import tqdm

from coolprompt.optimizer.hype.hype import HyPEOptimizer, Optimizer
from coolprompt.optimizer.hype.feedback_module import FeedbackModule
from coolprompt.utils.parsing import get_model_answer_extracted
from coolprompt.evaluator.evaluator import (
    Evaluator,
    EvalResultDetailed,
)


def sample_mini_batch(
    dataset: Sequence[str],
    targets: Sequence[str | int],
    size: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str | int]]:
    """Sample a mini-batch from the dataset.

    Returns:
        (samples, targets) - lists of length size (or less if dataset is smaller).
    """
    import random

    rng = random.Random(seed)
    n = min(size, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return (
        [dataset[i] for i in indices],
        [targets[i] for i in indices],
    )


def compute_pareto_front(
    candidates: List[str],
    results: List[EvalResultDetailed],
) -> List[Tuple[str, EvalResultDetailed]]:
    """Compute Pareto front from candidates based on score_per_task.

    A candidate dominates another if its score_per_task >= other.score_per_task
    for all tasks and > for at least one.

    Returns:
        List of (candidate, result) that belong to the Pareto front.
    """
    n = len(candidates)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Check if i dominates j
            i_scores = results[i].score_per_task
            j_scores = results[j].score_per_task
            if not i_scores or not j_scores:
                continue
            if len(i_scores) != len(j_scores):
                continue
            i_dominates_j = all(
                i_s >= j_s for i_s, j_s in zip(i_scores, j_scores)
            ) and any(i_s > j_s for i_s, j_s in zip(i_scores, j_scores))
            if i_dominates_j:
                is_pareto[j] = False

    return [(candidates[i], results[i]) for i in range(n) if is_pareto[i]]


class HyPEROptimizer(Optimizer):
    """HyPE with iterative refinement via evaluation-based recommendations."""

    def __init__(
        self,
        model: Any,
        evaluator: Evaluator,
        *,
        n_iterations: int = 5,
        patience: int = None,
        n_candidates: int = 3,
        top_n_candidates: int = 3,
        k_samples: int = 3,
        mini_batch_size: int = 16,
    ) -> None:
        super().__init__(model)
        self.hype_module = HyPEOptimizer(model)
        self.evaluator = evaluator
        self.feedback_module = FeedbackModule(model)
        self.n_iterations = n_iterations
        self.patience = patience
        self.n_candidates = n_candidates
        self.top_n_candidates = top_n_candidates
        self.k_samples = k_samples
        self.mini_batch_size = mini_batch_size

    def _get_variants_from_best(self, best_prompt: str, n_candidates: int) -> List[str]:
        paraphrase_prompt = f"""Generate an alternative version of the following prompt. The new version must:
- Use different words, sentence structure, and tone (e.g., more formal, casual, or creative).
- Preserve the original meaning, key details, and language.
- Vary in length: slightly shorter or longer (up to 10%).
- Feel natural and coherent.
- Output only the text of the alternative prompt, without any additional commentary or formatting.

Original prompt:
{best_prompt}

Alternative prompt:"""
        raw_result = get_model_answer_extracted(
            self.model, paraphrase_prompt, n=n_candidates, temperature=0.9
        )
        return [best_prompt] + [self._process_model_output(r) for r in raw_result]

    def _process_model_output(self, output: Any) -> str:
        return output if isinstance(output, str) else str(output)

    def optimize(
        self,
        prompt: str,
        dataset_split: Tuple[
            Sequence[str], Sequence[str], Sequence[str], Sequence[str]
        ],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate candidates, evaluate, update recommendations, repeat."""
        train_samples, val_samples, train_targets, val_targets = dataset_split
        best_prompt = prompt
        best_score = self.evaluator.evaluate(
            prompt,
            list(val_samples),
            list(val_targets),
            batch_size=50,
            show_progress=False,
        )
        patience_counter = 0

        for iteration in tqdm(range(self.n_iterations), desc="HyPER iterations"):
            # 1. Generate candidates from best_prompt
            candidates = self._get_variants_from_best(
                best_prompt, n_candidates=self.n_candidates
            )

            if not candidates:
                return best_prompt

            # 2. Mini-batch from train
            samples, sample_targets = sample_mini_batch(
                train_samples, train_targets, self.mini_batch_size
            )
            if not samples:
                continue

            # 3. Evaluate candidates on mini-batch via evaluate_detailed
            results: List[EvalResultDetailed] = [
                self.evaluator.evaluate_detailed(cand, samples, sample_targets)
                for cand in candidates
            ]

            # 4. Pareto front
            pareto_front = compute_pareto_front(candidates, results)

            # Fallback: if all candidates are in front, sort by aggregate_score
            if len(pareto_front) == len(candidates) and self.top_n_candidates < len(
                candidates
            ):
                scored = sorted(
                    zip(candidates, results),
                    key=lambda x: x[1].aggregate_score,
                    reverse=True,
                )
                pareto_front = scored[: self.top_n_candidates]

            if not pareto_front:
                continue

            # 5. Collect recommendations for all candidates from Pareto front
            all_recs: List[str] = []
            for cand_prompt, res in pareto_front:
                failed_sample = random.sample(
                    res.failed_examples,
                    min(self.k_samples, len(res.failed_examples)),
                )
                recs = self.feedback_module.generate_recommendations(
                    cand_prompt, failed_sample
                )
                all_recs.extend(recs)

            # Filter and update recommendations
            all_recs = self.feedback_module.filter_recommendations(all_recs)

            self.hype_module.update_section("recommendations", all_recs)

            # 6. For each candidate from Pareto front
            for cand_prompt, res in pareto_front:
                optimized_prompt = self.hype_module.optimize(
                    cand_prompt, meta_info=meta_info
                )

                val_score = self.evaluator.evaluate(
                    optimized_prompt,
                    list(val_samples),
                    list(val_targets),
                    batch_size=50,
                    show_progress=False,
                )

                if val_score > best_score:
                    best_score = val_score
                    best_prompt = optimized_prompt
                    patience_counter = 0
                else:
                    patience_counter += 1

            if self.patience and patience_counter >= self.patience:
                break

        return best_prompt
