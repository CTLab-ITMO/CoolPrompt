"""HyPEROptimizer: HyPE with iterative refinement via recommendations."""

import logging
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

logger = logging.getLogger(__name__)


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
    ) -> Tuple[str, list]:
        """Generate candidates, evaluate, update recommendations, repeat."""
        train_samples, val_samples, train_targets, val_targets = dataset_split
        best_prompt = prompt
        logger.debug(f"[HyPER] Initial prompt: {prompt[:250]}...")
        
        best_score = self.evaluator.evaluate(
            prompt,
            list(val_samples),
            list(val_targets),
        )
        logger.debug(f"[HyPER] Initial validation score: {best_score}")
        
        patience_counter = 0
        iteration_history = []

        for iteration in tqdm(range(self.n_iterations), desc="HyPER iterations"):
            logger.debug(f"\n{'='*60}")
            logger.debug(f"[HyPER] Iteration {iteration + 1}/{self.n_iterations}")
            score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            logger.debug(f"[HyPER] Current best score: {score_str}")
            logger.debug(f"[HyPER] Current best prompt: {best_prompt[:250]}...")
            logger.debug(f"{'='*60}")
            
            score_before_iteration = best_score
            
            # 1. Generate candidates from best_prompt
            logger.debug(f"[HyPER] Generating {self.n_candidates} candidates...")
            candidates = self._get_variants_from_best(
                best_prompt, n_candidates=self.n_candidates
            )
            logger.debug(f"[HyPER] Generated {len(candidates)} candidates")

            if not candidates:
                logger.warning("[HyPER] No candidates generated, returning best prompt")
                return best_prompt, iteration_history

            # 2. Mini-batch from train
            samples, sample_targets = sample_mini_batch(
                train_samples, train_targets, self.mini_batch_size
            )
            logger.debug(f"[HyPER] Sampled {len(samples)} mini-batch samples")
            
            if not samples:
                logger.warning("[HyPER] No samples in mini-batch, skipping iteration")
                continue

            # 3. Evaluate candidates on mini-batch
            logger.debug(f"[HyPER] Evaluating {len(candidates)} candidates on mini-batch, k_samples={self.k_samples}...")
            results: List[EvalResultDetailed] = [
                self.evaluator.evaluate(cand, samples, sample_targets, failed_examples=self.k_samples, return_detailed=True)
                for cand in candidates
            ]
            
            # Log each candidate's score in debug mode
            for i, (cand, res) in enumerate(zip(candidates, results)):
                score_str = f"{res.aggregate_score:.4f}" if res.aggregate_score is not None else "N/A"
                logger.debug(f"[HyPER] Candidate {i+1}: score={score_str}, failed={len(res.failed_examples)}")

            # 4. Pareto front
            pareto_front = compute_pareto_front(candidates, results)
            logger.debug(f"[HyPER] Pareto front size: {len(pareto_front)}")

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
                logger.debug(f"[HyPER] Using top {self.top_n_candidates} by score")

            if not pareto_front:
                logger.warning("[HyPER] Empty Pareto front, skipping iteration")
                continue

            # 5. Collect recommendations for all candidates from Pareto front
            logger.debug(f"[HyPER] Collecting recommendations from {len(pareto_front)} Pareto candidates...")
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
            logger.debug(f"[HyPER] Generated {len(all_recs)} recommendations")

            self.hype_module.update_section("recommendations", all_recs)

            # 6. For each candidate from Pareto front
            logger.debug(f"[HyPER] Evaluating {len(pareto_front)} optimized candidates on validation set...")
            for i, (cand_prompt, res) in enumerate(pareto_front):
                logger.debug(f"[HyPER] Optimizing Pareto candidate {i+1}/{len(pareto_front)}...")
                
                optimized_prompt = self.hype_module.optimize(
                    cand_prompt, meta_info=meta_info
                )

                val_score = self.evaluator.evaluate(
                    optimized_prompt,
                    list(val_samples),
                    list(val_targets),
                )
                
                val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
                logger.debug(f"[HyPER] Optimized candidate {i+1} validation score: {val_score_str}")

                if val_score is not None and best_score is not None and val_score > best_score:
                    old_str = f"{best_score:.4f}"
                    new_str = f"{val_score:.4f}"
                    logger.debug(f"[HyPER] *** NEW BEST! Score improved: {old_str} -> {new_str}")
                    best_score = val_score
                    best_prompt = optimized_prompt
            
            if best_score == score_before_iteration:
                patience_counter += 1
                logger.debug(f"[HyPER] No improvement in iteration {iteration + 1}, patience: {patience_counter}/{self.patience}")
            else:
                patience_counter = 0
            
            iteration_history.append({
                "iteration": iteration + 1,
                "best_score": best_score,
                "patience_counter": patience_counter,
            })
            score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            logger.debug(f"[HyPER] Iteration {iteration + 1} complete. Best score: {score_str}")

            if self.patience and patience_counter >= self.patience:
                logger.debug(f"[HyPER] Early stopping: patience {self.patience} reached")
                break

        logger.debug(f"\n{'='*60}")
        logger.debug(f"[HyPER] Optimization complete!")
        final_score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        logger.debug(f"[HyPER] Final best score: {final_score_str}")
        logger.debug(f"[HyPER] Final best prompt: {best_prompt[:200]}...")
        logger.debug(f"[HyPER] Iteration history: {iteration_history}")
        logger.debug(f"{'='*60}")
        
        return best_prompt, iteration_history
