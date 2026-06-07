"""Compact HyPER implementation for producing the optimized prompt only."""

from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from coolprompt.evaluator.evaluator import EvalResultDetailed, Evaluator
from coolprompt.evaluator.metrics import BertScoreMetric
from coolprompt.optimizer.hyper.feedback_module import (
    ContrastiveCandidate,
    FeedbackModule,
)
from coolprompt.optimizer.hyper.meta_prompt import MetaPromptOptimizer
from coolprompt.utils.parsing import get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import PARAPHRASE_PROMPT


_BERTSCORE_MODEL_TYPE = "microsoft/deberta-large-mnli"
_bertscore_evaluate = None


def _get_bertscore_evaluate(metric: Any):
    if isinstance(metric, BertScoreMetric):
        return metric._metric

    global _bertscore_evaluate
    if _bertscore_evaluate is None:
        from evaluate import load

        _bertscore_evaluate = load("bertscore")
    return _bertscore_evaluate


def sample_mini_batch_with_indices(
    dataset: Sequence[str],
    targets: Sequence[str | int],
    size: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str | int], List[int]]:
    """Sample a mini-batch and return its source indices."""
    rng = random.Random(seed)
    n = min(size, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return (
        [dataset[i] for i in indices],
        [targets[i] for i in indices],
        indices,
    )


def _compute_similarity_matrix(prompts: List[str], bertscore_evaluate: Any) -> np.ndarray:
    n_prompts = len(prompts)
    similarity = np.eye(n_prompts)

    pairs = [(i, j) for i in range(n_prompts) for j in range(i + 1, n_prompts)]
    if not pairs:
        return similarity

    result = bertscore_evaluate.compute(
        predictions=[prompts[i] for i, _ in pairs],
        references=[prompts[j] for _, j in pairs],
        model_type=_BERTSCORE_MODEL_TYPE,
    )

    for idx, (i, j) in enumerate(pairs):
        similarity[i, j] = result["f1"][idx]
        similarity[j, i] = result["f1"][idx]

    return similarity


def _adaptive_lambda(best_score: float) -> float:
    min_lambda, max_lambda = 0.5, 0.9
    return max_lambda - (max_lambda - min_lambda) * best_score


def mmr_select(
    candidates: List[str],
    results: List[EvalResultDetailed],
    top_n: int,
    lambda_: float,
    bertscore_evaluate: Any,
) -> List[Tuple[str, EvalResultDetailed]]:
    """Select candidates with maximal marginal relevance."""
    if len(candidates) <= top_n:
        return list(zip(candidates, results))

    similarity = _compute_similarity_matrix(candidates, bertscore_evaluate)
    scores = np.array(
        [result.aggregate_score or 0.0 for result in results],
        dtype=float,
    )

    selected = [int(np.argmax(scores))]
    remaining = [idx for idx in range(len(candidates)) if idx not in selected]

    while len(selected) < top_n and remaining:
        mmr_scores = {
            idx: lambda_ * scores[idx]
            - (1 - lambda_)
            * max(similarity[idx, selected_idx] for selected_idx in selected)
            for idx in remaining
        }
        next_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_idx)
        remaining.remove(next_idx)

    return [(candidates[idx], results[idx]) for idx in selected]


class HyPERSubmitOptimizer:
    """Standalone HyPER optimizer used by submission-style entrypoints."""

    def __init__(
        self,
        model: Any,
        evaluator: Evaluator,
        *,
        n_iterations: int = 5,
        patience: int | None = None,
        n_candidates: int = 3,
        top_n_candidates: int = 3,
        k_samples: int = 3,
        mini_batch_size: int = 16,
        contrastive_probability: float = 0.5,
        contrastive_max_answer_chars: int = 500,
        feedback_answer_head_chars: int = 500,
        feedback_answer_tail_chars: int = 500,
        enable_instance_leak_audit: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize submission HyPER search parameters."""
        self.model = model
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        self.patience = patience
        self.n_candidates = n_candidates
        self.top_n_candidates = top_n_candidates
        self.k_samples = k_samples
        self.mini_batch_size = mini_batch_size
        self.enable_instance_leak_audit = enable_instance_leak_audit
        self.random_seed = random_seed

        self.meta_prompt_module = MetaPromptOptimizer(model)
        self.feedback_module = FeedbackModule(
            model,
            section_specs=self.meta_prompt_module.builder.config.section_specs,
            contrastive_probability=contrastive_probability,
            contrastive_max_answer_chars=contrastive_max_answer_chars,
            feedback_answer_head_chars=feedback_answer_head_chars,
            feedback_answer_tail_chars=feedback_answer_tail_chars,
        )

    def _get_variants_from_best(self, best_prompt: str) -> List[str]:
        raw_result = get_model_answer_extracted(
            self.model,
            PARAPHRASE_PROMPT.format(prompt=best_prompt),
            n=self.n_candidates,
            temperature=0.9,
        )
        return [best_prompt] + [str(result) for result in raw_result]

    def optimize(
        self,
        prompt: str,
        dataset_split: Tuple[
            Sequence[str], Sequence[str], Sequence[str | int], Sequence[str | int]
        ],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run submission HyPER optimization and return the best prompt."""
        train_samples, val_samples, train_targets, val_targets = dataset_split

        best_prompt = prompt
        best_score = self.evaluator.evaluate(prompt, list(val_samples), list(val_targets))
        patience_counter = 0

        for iteration in range(self.n_iterations):
            score_before_iteration = best_score
            best_prompt_before_iteration = best_prompt

            candidates = self._get_variants_from_best(best_prompt)
            if not candidates:
                return best_prompt

            mini_batch_seed = (
                None if self.random_seed is None else self.random_seed + iteration
            )
            samples, sample_targets, _ = sample_mini_batch_with_indices(
                train_samples,
                train_targets,
                self.mini_batch_size,
                seed=mini_batch_seed,
            )
            if not samples:
                continue

            results = [
                self.evaluator.evaluate(
                    candidate,
                    samples,
                    sample_targets,
                    failed_examples=self.k_samples,
                    return_detailed=True,
                )
                for candidate in candidates
            ]

            if sum(len(result.failed_examples) for result in results) == 0:
                resample_seed = (
                    None
                    if self.random_seed is None
                    else self.random_seed + 10_000 + iteration
                )
                samples, sample_targets, _ = sample_mini_batch_with_indices(
                    train_samples,
                    train_targets,
                    self.mini_batch_size,
                    seed=resample_seed,
                )
                results = [
                    self.evaluator.evaluate(
                        candidate,
                        samples,
                        sample_targets,
                        failed_examples=self.k_samples,
                        return_detailed=True,
                    )
                    for candidate in candidates
                ]

            bertscore_evaluate = _get_bertscore_evaluate(self.evaluator.metric)
            lambda_ = _adaptive_lambda(best_score or 0.0)
            selected = mmr_select(
                candidates,
                results,
                self.top_n_candidates,
                lambda_,
                bertscore_evaluate,
            )
            if not selected:
                continue

            selected_prompts = {candidate for candidate, _ in selected}
            feedback_sources = [
                (candidate, result)
                for candidate, result in selected
                if result.failed_examples
            ]

            substitutes_needed = self.top_n_candidates - len(feedback_sources)
            if substitutes_needed > 0:
                substitutes = sorted(
                    [
                        (candidate, result)
                        for candidate, result in zip(candidates, results)
                        if candidate not in selected_prompts and result.failed_examples
                    ],
                    key=lambda item: item[1].aggregate_score or 0.0,
                    reverse=True,
                )[:substitutes_needed]
                feedback_sources.extend(substitutes)

            recommendations = []
            for candidate_prompt, result in feedback_sources:
                failed_sample = random.sample(
                    result.failed_examples,
                    min(self.k_samples, len(result.failed_examples)),
                )

                contrastive_per_failure = []
                for failed_example in failed_sample:
                    alternatives = []
                    if failed_example.batch_index >= 0:
                        for other_candidate, other_result in zip(candidates, results):
                            if other_candidate == candidate_prompt:
                                continue
                            if not other_result.score_per_task or not other_result.raw_outputs:
                                continue
                            if failed_example.batch_index >= len(
                                other_result.score_per_task
                            ):
                                continue

                            raw_answer = other_result.raw_outputs[
                                failed_example.batch_index
                            ]
                            alternatives.append(
                                ContrastiveCandidate(
                                    prompt=other_candidate,
                                    score=float(
                                        other_result.score_per_task[
                                            failed_example.batch_index
                                        ]
                                    ),
                                    raw_answer=raw_answer,
                                    parsed_answer=self.evaluator.metric.parse_output(
                                        raw_answer
                                    ),
                                )
                            )
                    contrastive_per_failure.append(alternatives)

                recommendations.extend(
                    self.feedback_module.generate_recommendations(
                        candidate_prompt,
                        failed_sample,
                        contrastive_candidates_per_failure=contrastive_per_failure,
                    )
                )

            if recommendations:
                recommendations = self.feedback_module.filter_recommendations(
                    recommendations
                )
                problem_description = (meta_info or {}).get("problem_description", "")
                if (
                    self.enable_instance_leak_audit
                    and problem_description
                    and recommendations
                ):
                    recommendations = self.feedback_module.drop_instance_leaks(
                        recommendations,
                        problem_description,
                    )
                self.meta_prompt_module.update_section("recommendations", recommendations)

            validation_score_cache = {
                best_prompt_before_iteration: score_before_iteration,
            }
            for candidate_prompt, _ in selected:
                optimized_prompt = self.meta_prompt_module.optimize(
                    candidate_prompt,
                    meta_info=meta_info,
                )

                if optimized_prompt in validation_score_cache:
                    validation_score = validation_score_cache[optimized_prompt]
                else:
                    validation_score = self.evaluator.evaluate(
                        optimized_prompt,
                        list(val_samples),
                        list(val_targets),
                    )
                    validation_score_cache[optimized_prompt] = validation_score

                if (
                    validation_score is not None
                    and best_score is not None
                    and validation_score > best_score
                ):
                    best_score = validation_score
                    best_prompt = optimized_prompt

            if best_score == score_before_iteration:
                patience_counter += 1
            else:
                patience_counter = 0

            if self.patience and patience_counter >= self.patience:
                break

        return best_prompt


def get_hyper_prompt(
    model: Any,
    evaluator: Evaluator,
    initial_prompt: str,
    dataset_split: Tuple[
        Sequence[str], Sequence[str], Sequence[str | int], Sequence[str | int]
    ],
    meta_info: Optional[dict[str, Any]] = None,
    **optimizer_kwargs: Any,
) -> str:
    """Build and run ``HyPERSubmitOptimizer`` for a single prompt."""
    optimizer = HyPERSubmitOptimizer(
        model=model,
        evaluator=evaluator,
        **optimizer_kwargs,
    )
    return optimizer.optimize(
        prompt=initial_prompt,
        dataset_split=dataset_split,
        meta_info=meta_info,
    )
