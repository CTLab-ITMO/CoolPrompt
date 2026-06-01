"""Iterative HyPER optimizer: meta-prompt refinement driven by evaluation feedback."""

from __future__ import annotations

import logging
import random
from typing import Any, List, Optional, Sequence, Tuple, override

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)

import numpy as np
from tqdm import tqdm

from coolprompt.optimizer.hyper.meta_prompt import MetaPromptOptimizer, Optimizer
from coolprompt.optimizer.hyper.feedback_module import (
    ContrastiveCandidate,
    FeedbackModule,
)
from coolprompt.utils.parsing import get_model_answer_extracted
from coolprompt.evaluator.evaluator import (
    Evaluator,
    EvalResultDetailed,
)
from coolprompt.evaluator.metrics import BertScoreMetric
from coolprompt.utils.prompt_templates.hyper_templates import (
    PARAPHRASE_PROMPT,
    Recommendation,
)
from coolprompt.optimizer.structured_schemas.hyper import (
    ParaphrasedVariantResponse,
)

_BERTSCORE_MODEL_TYPE = "microsoft/deberta-large-mnli"
_bertscore_evaluate = None  # module-level singleton


def _get_bertscore_evaluate(metric: Any):
    """Return a HuggingFace ``evaluate`` handle for BERTScore F1.

    Args:
        metric: Evaluator metric; if it is a :class:`BertScoreMetric`, reuse its
            internal ``evaluate`` module instance.

    Returns:
        Loaded ``bertscore`` metric object suitable for ``.compute(...)``.
    """
    if isinstance(metric, BertScoreMetric):
        return metric._metric

    global _bertscore_evaluate
    if _bertscore_evaluate is None:
        from evaluate import load
        _bertscore_evaluate = load("bertscore")
    return _bertscore_evaluate

logger = logging.getLogger(__name__)


def sample_mini_batch(
    dataset: Sequence[str],
    targets: Sequence[str | int],
    size: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str | int]]:
    """Draw a random mini-batch without replacement from parallel lists.

    Args:
        dataset: Input strings (same length as ``targets``).
        targets: Parallel labels or targets.
        size: Requested batch size (capped by dataset length).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tuple ``(samples, targets)`` of equal length (at most ``size``).
    """
    import random

    rng = random.Random(seed)
    n = min(size, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return (
        [dataset[i] for i in indices],
        [targets[i] for i in indices],
    )


def sample_mini_batch_with_indices(
    dataset: Sequence[str],
    targets: Sequence[str | int],
    size: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str | int], List[int]]:
    """Same as :func:`sample_mini_batch` but also returns chosen row indices.

    Args:
        dataset: Input strings.
        targets: Parallel targets.
        size: Requested batch size.
        seed: Optional RNG seed.

    Returns:
        Tuple ``(samples, targets, indices)`` where ``indices`` index into the
        original ``dataset`` / ``targets`` lists.
    """
    rng = random.Random(seed)
    n = min(size, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return (
        [dataset[i] for i in indices],
        [targets[i] for i in indices],
        indices,
    )


def _compute_similarity_matrix(prompts: List[str], bertscore_evaluate: Any) -> np.ndarray:
    """Pairwise semantic similarity (BERTScore F1) between prompt strings.

    Args:
        prompts: Candidate prompt texts (length ``K``).
        bertscore_evaluate: Loaded ``evaluate`` bertscore module.

    Returns:
        ``K x K`` symmetric matrix with ones on the diagonal; upper triangle filled
        from unique unordered pairs only.
    """
    K = len(prompts)
    sim_matrix = np.eye(K)

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    if not pairs:
        return sim_matrix

    result = bertscore_evaluate.compute(
        predictions=[prompts[i] for i, _ in pairs],
        references=[prompts[j] for _, j in pairs],
        model_type=_BERTSCORE_MODEL_TYPE,
    )

    for idx, (i, j) in enumerate(pairs):
        sim_matrix[i, j] = result["f1"][idx]
        sim_matrix[j, i] = result["f1"][idx]

    return sim_matrix


def _adaptive_lambda(p_star_score: float) -> float:
    """Map validation ``p*`` to MMR diversity weight ``lambda`` in ``[0.5, 0.9]``.

    Args:
        p_star_score: Current best aggregate score in ``[0, 1]`` (clamped implicitly).

    Returns:
        ``lambda`` for MMR: higher best score gives more diversity (higher lambda).
    """
    min_lambda, max_lambda = 0.5, 0.9
    return max_lambda - (max_lambda - min_lambda) * p_star_score


def mmr_select(
    candidates: List[str],
    results: List[EvalResultDetailed],
    top_n: int,
    lambda_: float,
    bertscore_evaluate: Any,
) -> List[Tuple[str, EvalResultDetailed]]:
    """Select ``top_n`` candidates by maximal marginal relevance (MMR).

    Uses ``MMR(d) = λ·score(d) − (1−λ)·max_{j in selected} sim(d, d_j)`` with
    BERTScore-based ``sim``.

    Args:
        candidates: Paraphrase prompt strings (parallel to ``results``).
        results: Detailed evaluation rows for each candidate.
        top_n: Number of items to keep.
        lambda_: Trade-off between relevance and diversity (see :func:`_adaptive_lambda`).
        bertscore_evaluate: BERTScore ``evaluate`` handle.

    Returns:
        List of ``(prompt, EvalResultDetailed)`` pairs of length ``min(top_n, K)``.
    """
    if len(candidates) <= top_n:
        return list(zip(candidates, results))

    sim_matrix = _compute_similarity_matrix(candidates, bertscore_evaluate)

    scores = np.array([
        r.aggregate_score if r.aggregate_score is not None else 0.0
        for r in results
    ])

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    # Seed selection with the highest-scoring candidate.
    best_idx = int(np.argmax(scores))
    selected.append(best_idx)
    remaining.remove(best_idx)

    while len(selected) < top_n and remaining:
        mmr_scores = {
            idx: lambda_ * scores[idx]
                 - (1 - lambda_) * max(sim_matrix[idx, j] for j in selected)
            for idx in remaining
        }
        next_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_idx)
        remaining.remove(next_idx)

    return [(candidates[i], results[i]) for i in selected]


class HyPEROptimizer(Optimizer):
    """HyPER: outer loop of candidates, feedback, and inner :class:`MetaPromptOptimizer`."""

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
        contrastive_probability: float = 0.5,
        contrastive_max_answer_chars: int = 500,
        feedback_answer_head_chars: int = 500,
        feedback_answer_tail_chars: int = 500,
        enable_instance_leak_audit: bool = True,
        random_seed: Optional[int] = None,
        use_structured_output: bool = False,
    ) -> None:
        """Configure HyPER hyperparameters and construct submodules.

        Args:
            model: Chat model for paraphrases, feedback, and meta-prompt calls.
            evaluator: Task evaluator (mini-batch + validation scores).
            n_iterations: Maximum outer-loop iterations.
            patience: Optional early-stop patience on validation non-improvement.
            n_candidates: Paraphrase count per iteration (excluding the original).
            top_n_candidates: MMR shortlist size for inner optimization.
            k_samples: Max failed examples sampled per feedback source.
            mini_batch_size: Train mini-batch size for candidate scoring.
            contrastive_probability: Bernoulli probability to try contrastive feedback.
            contrastive_max_answer_chars: Budget for contrastive winning answers.
            feedback_answer_head_chars: Head truncation for failure answers in feedback.
            feedback_answer_tail_chars: Tail truncation for failure answers in feedback.
            enable_instance_leak_audit: If True, run ``drop_instance_leaks`` when
                ``meta_info`` contains a non-empty ``problem_description``.
            random_seed: Base seed for mini-batch sampling (per-iteration offset applied).
        """
        super().__init__(model)
        self.use_structured_output = use_structured_output
        self.meta_prompt_module = MetaPromptOptimizer(
            model, use_structured_output=use_structured_output
        )
        self.evaluator = evaluator
        self.contrastive_probability = contrastive_probability
        self.contrastive_max_answer_chars = contrastive_max_answer_chars
        self.feedback_answer_head_chars = feedback_answer_head_chars
        self.feedback_answer_tail_chars = feedback_answer_tail_chars
        self.enable_instance_leak_audit = enable_instance_leak_audit
        # Feedback module knows about the resulting-prompt sections so that
        # recommendations can be section-targeted (or 'general').
        self.feedback_module = FeedbackModule(
            model,
            section_specs=self.meta_prompt_module.builder.config.section_specs,
            contrastive_probability=contrastive_probability,
            contrastive_max_answer_chars=contrastive_max_answer_chars,
            feedback_answer_head_chars=feedback_answer_head_chars,
            feedback_answer_tail_chars=feedback_answer_tail_chars,
            use_structured_output=use_structured_output,
        )
        self.n_iterations = n_iterations
        self.patience = patience
        self.n_candidates = n_candidates
        self.top_n_candidates = top_n_candidates
        self.k_samples = k_samples
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed

    def _get_variants_from_best(self, best_prompt: str, n_candidates: int) -> List[str]:
        """Paraphrase ``best_prompt`` and prepend the unchanged original.

        Args:
            best_prompt: Current best prompt text.
            n_candidates: Number of paraphrases requested from the LLM.

        Returns:
            List whose first element is ``best_prompt`` followed by paraphrases.
        """
        query = PARAPHRASE_PROMPT.format(prompt=best_prompt)
        if self.use_structured_output:
            structured = self.model.bind(temperature=0.9).with_structured_output(
                ParaphrasedVariantResponse, method="json_schema"
            )
            raw_outputs = [
                r.paraphrased_prompt for r in structured.batch([query] * n_candidates)
            ]
            raw_outputs = list(dict.fromkeys(raw_outputs))
            return [best_prompt] + raw_outputs
        raw_result = get_model_answer_extracted(
            self.model, query, n=n_candidates, temperature=0.9
        )
        return [best_prompt] + [self._process_model_output(r) for r in raw_result]

    def _process_model_output(self, output: Any) -> str:
        """Normalize a single model return value to ``str``."""
        return output if isinstance(output, str) else str(output)

    def optimize(
        self,
        prompt: str,
        dataset_split: Tuple[
            Sequence[str], Sequence[str], Sequence[str], Sequence[str]
        ],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list]:
        """Run the full HyPER outer loop over train/val splits.

        Args:
            prompt: Initial prompt to optimize.
            dataset_split: Tuple ``(train_x, val_x, train_y, val_y)`` of iterables.
            meta_info: Optional task metadata; ``problem_description`` enables leak audit.

        Returns:
            Tuple ``(best_prompt, iteration_history)`` where ``iteration_history`` is a
            list of per-iteration serializable dict records.
        """
        train_samples, val_samples, train_targets, val_targets = dataset_split

        best_prompt = prompt
        logger.info("[HyPER] Starting optimization")
        logger.debug(f"[HyPER] Initial prompt: {prompt[:250]}...")
        best_score = self.evaluator.evaluate(
            prompt,
            list(val_samples),
            list(val_targets),
        )
        logger.info(
            f"[HyPER] Initial validation score: {best_score:.4f}"
            if best_score is not None
            else "[HyPER] Initial validation score: N/A"
        )
        patience_counter = 0
        iteration_history: list = []
        start_iteration = 0

        logger.info(
            f"[HyPER] Config: n_iterations={self.n_iterations}, patience={self.patience}, "
            f"n_candidates={self.n_candidates}, mini_batch_size={self.mini_batch_size}, "
            f"k_samples={self.k_samples}, top_n_candidates={self.top_n_candidates}"
        )

        for iteration in tqdm(
            range(start_iteration, self.n_iterations),
            desc="HyPER iterations",
            initial=start_iteration,
            total=self.n_iterations,
        ):
            score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            logger.info(f"\n{'='*60}")
            logger.info(f"[HyPER] === Iteration {iteration + 1}/{self.n_iterations} | best_score={score_str} ===")
            logger.debug(f"[HyPER] Current best prompt: {best_prompt[:250]}...")

            score_before_iteration = best_score
            best_prompt_before_iteration = best_prompt

            # 1. Generate candidates from best_prompt
            logger.info(f"[HyPER] Step 1: Generating {self.n_candidates} paraphrase candidates...")
            candidates = self._get_variants_from_best(
                best_prompt, n_candidates=self.n_candidates
            )
            logger.info(f"[HyPER] Generated {len(candidates)} candidates (including original)")

            if not candidates:
                logger.warning("[HyPER] No candidates generated, returning best prompt")
                return best_prompt, iteration_history

            # 2. Mini-batch from train. Validation is fixed for the whole run;
            # mini-batches are deterministic when random_seed is provided.
            mini_batch_seed = (
                None if self.random_seed is None else self.random_seed + iteration
            )
            samples, sample_targets, mini_batch_indices = sample_mini_batch_with_indices(
                train_samples, train_targets, self.mini_batch_size, seed=mini_batch_seed
            )
            logger.info(
                f"[HyPER] Step 2: Mini-batch sampled: {len(samples)} examples "
                f"(seed={mini_batch_seed}, indices={mini_batch_indices})"
            )

            if not samples:
                logger.warning("[HyPER] No samples in mini-batch, skipping iteration")
                continue

            # 3. Evaluate candidates on mini-batch
            logger.info(f"[HyPER] Step 3: Evaluating {len(candidates)} candidates on mini-batch...")
            results: List[EvalResultDetailed] = [
                self.evaluator.evaluate(cand, samples, sample_targets, failed_examples=self.k_samples, return_detailed=True)
                for cand in candidates
            ]

            for i, (cand, res) in enumerate(zip(candidates, results)):
                score_str = f"{res.aggregate_score:.4f}" if res.aggregate_score is not None else "N/A"
                logger.info(f"[HyPER]   Candidate {i+1}: mini_batch_score={score_str}, n_examples_for_analysis={len(res.failed_examples)}")
                logger.debug(f"[HyPER]   Candidate {i+1} prompt:\n{cand}")

            # 3.5. Guard: if no candidate has failures, mini-batch is too easy; resample once.
            total_failures = sum(len(r.failed_examples) for r in results)
            mini_batch_resampled = False
            mini_batch_resample_seed = None
            if total_failures == 0:
                logger.info(
                    f"[HyPER] Step 3.5: All candidates have 0 failures on mini-batch. Resampling once..."
                )
                mini_batch_resampled = True
                mini_batch_resample_seed = (
                    None if self.random_seed is None else self.random_seed + 10_000 + iteration
                )
                samples, sample_targets, mini_batch_indices = sample_mini_batch_with_indices(
                    train_samples, train_targets, self.mini_batch_size, seed=mini_batch_resample_seed
                )
                logger.info(
                    f"[HyPER]   Resampled mini-batch: {len(samples)} examples "
                    f"(seed={mini_batch_resample_seed}, indices={mini_batch_indices})"
                )
                results = [
                    self.evaluator.evaluate(
                        cand, samples, sample_targets,
                        failed_examples=self.k_samples, return_detailed=True,
                    )
                    for cand in candidates
                ]
                total_failures = sum(len(r.failed_examples) for r in results)
                logger.info(f"[HyPER]   After resample: total_failures={total_failures}")

            # 4. MMR selection (relevance + diversity)
            bertscore_evaluate = _get_bertscore_evaluate(self.evaluator.metric)
            lambda_ = _adaptive_lambda(best_score if best_score is not None else 0.0)
            selected = mmr_select(
                candidates=candidates,
                results=results,
                top_n=self.top_n_candidates,
                lambda_=lambda_,
                bertscore_evaluate=bertscore_evaluate,
            )
            p_star_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            logger.info(
                f"[HyPER] Step 4: MMR selected {len(selected)}/{len(candidates)} "
                f"(λ={lambda_:.2f}, p_star_score={p_star_str})"
            )

            if not selected:
                logger.warning("[HyPER] No candidates selected by MMR, skipping iteration")
                continue

            for i, (cand_prompt, res) in enumerate(selected):
                score_str = f"{res.aggregate_score:.4f}" if res.aggregate_score is not None else "N/A"
                logger.info(f"[HyPER]   MMR[{i+1}] mini_batch_score={score_str}")
                logger.debug(f"[HyPER]   MMR[{i+1}] prompt:\n{cand_prompt}")

            # 5. Build feedback sources: top_n_candidates contributors with failures.
            # Selected candidates with failures go first; selected without failures
            # are substituted with non-selected candidates that have failures
            # (sorted by score desc).
            selected_prompts = {cand for cand, _ in selected}
            feedback_sources: List[Tuple[str, EvalResultDetailed]] = [
                (cand, res) for cand, res in selected if res.failed_examples
            ]
            substitutes_needed = self.top_n_candidates - len(feedback_sources)
            if substitutes_needed > 0:
                substitutes = sorted(
                    [
                        (cand, res)
                        for cand, res in zip(candidates, results)
                        if cand not in selected_prompts and res.failed_examples
                    ],
                    key=lambda x: x[1].aggregate_score if x[1].aggregate_score is not None else 0.0,
                    reverse=True,
                )[:substitutes_needed]
                feedback_sources.extend(substitutes)
                if substitutes:
                    logger.info(
                        f"[HyPER]   Substituted {len(substitutes)} feedback source(s) "
                        f"from non-selected candidates"
                    )

            logger.info(
                f"[HyPER] Step 5: Generating recommendations from {len(feedback_sources)} "
                f"source(s) (target={self.top_n_candidates})..."
            )
            all_recs: List[Recommendation] = []
            for cand_prompt, res in feedback_sources:
                failed_sample = random.sample(
                    res.failed_examples,
                    min(self.k_samples, len(res.failed_examples)),
                )
                # For each failure, build the list of OTHER candidates' performance
                # on the SAME mini-batch index — used by feedback module for
                # contrastive recommendations (when coin flips heads).
                contrastive_per_failure: List[List[ContrastiveCandidate]] = []
                for fe in failed_sample:
                    others: List[ContrastiveCandidate] = []
                    if fe.batch_index >= 0:
                        for other_cand, other_res in zip(candidates, results):
                            if other_cand == cand_prompt:
                                continue
                            if not other_res.score_per_task or not other_res.raw_outputs:
                                continue
                            if fe.batch_index >= len(other_res.score_per_task):
                                continue
                            others.append(
                                ContrastiveCandidate(
                                    prompt=other_cand,
                                    score=float(other_res.score_per_task[fe.batch_index]),
                                    raw_answer=other_res.raw_outputs[fe.batch_index],
                                    parsed_answer=self.evaluator.metric.parse_output(
                                        other_res.raw_outputs[fe.batch_index]
                                    ),
                                )
                            )
                    contrastive_per_failure.append(others)

                recs = self.feedback_module.generate_recommendations(
                    cand_prompt,
                    failed_sample,
                    contrastive_candidates_per_failure=contrastive_per_failure,
                )
                all_recs.extend(recs)

            if all_recs:
                all_recs = self.feedback_module.filter_recommendations(all_recs)
                recommendations_before_audit = list(all_recs)
                audit_trace = []
                logger.info(f"[HyPER]   {len(all_recs)} recommendations after filtering:")
                for i, rec in enumerate(all_recs):
                    logger.info(f"[HyPER]   Rec {i+1} [{rec.section}]: {rec.text}")

                # Audit instance leaks: keep broad recs, rewrite useful leaky
                # recs, and drop narrow/vague ones (one LLM call).
                problem_description = (meta_info or {}).get("problem_description", "")
                if (
                    self.enable_instance_leak_audit
                    and problem_description
                    and all_recs
                ):
                    pre_audit = len(all_recs)
                    all_recs = self.feedback_module.drop_instance_leaks(
                        all_recs, problem_description
                    )
                    audit_trace = list(self.feedback_module.last_audit_trace)
                    rewritten = sum(
                        1
                        for t in audit_trace
                        if str(t.get("verdict", "")).upper().startswith("REWRITE")
                    )
                    dropped = pre_audit - len(all_recs)
                    logger.info(
                        f"[HyPER]   Audit: rewritten {rewritten}, "
                        f"dropped {dropped}/{pre_audit} instance-leak recs "
                        f"({len(all_recs)} kept)"
                    )
                    for i, rec in enumerate(all_recs):
                        logger.info(f"[HyPER]   AuditedRec {i+1} [{rec.section}]: {rec.text}")

                self.meta_prompt_module.update_section("recommendations", all_recs)
            else:
                recommendations_before_audit = []
                audit_trace = []
                logger.warning(
                    "[HyPER]   No feedback sources with failures available; "
                    "keeping previous recommendations unchanged"
                )

            # 6. Meta-prompt optimization for each MMR-selected candidate; validate on val.
            logger.info(f"[HyPER] Step 6: MetaPrompt-optimizing {len(selected)} MMR candidates + val evaluation...")
            validation_score_cache = {best_prompt_before_iteration: score_before_iteration}
            optimized_val_scores: List[dict] = []
            for i, (cand_prompt, res) in enumerate(selected):
                logger.info(f"[HyPER]   Optimizing MMR[{i+1}] with MetaPrompt...")
                optimized_prompt = self.meta_prompt_module.optimize(
                    cand_prompt, meta_info=meta_info
                )
                logger.debug(f"[HyPER]   MMR[{i+1}] optimized prompt:\n{optimized_prompt}")

                val_score_cached = optimized_prompt in validation_score_cache
                if val_score_cached:
                    val_score = validation_score_cache[optimized_prompt]
                    logger.info(
                        f"[HyPER]   MMR[{i+1}] validation cache hit "
                        "(prompt already scored)"
                    )
                else:
                    val_score = self.evaluator.evaluate(
                        optimized_prompt,
                        list(val_samples),
                        list(val_targets),
                    )
                    validation_score_cache[optimized_prompt] = val_score
                val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
                cached_suffix = " cached" if val_score_cached else ""
                logger.info(f"[HyPER]   MMR[{i+1}] val_score={val_score_str}{cached_suffix}")
                optimized_val_scores.append(
                    {
                        "prompt": optimized_prompt,
                        "val_score": val_score,
                        "val_score_cached": val_score_cached,
                    }
                )

                if val_score is not None and best_score is not None and val_score > best_score:
                    logger.info(f"[HyPER]   *** NEW BEST: {best_score:.4f} -> {val_score:.4f} (MMR[{i+1}])")
                    best_score = val_score
                    best_prompt = optimized_prompt

            if best_score == score_before_iteration:
                patience_counter += 1
                logger.info(
                    f"[HyPER] No improvement this iteration. "
                    f"patience_counter={patience_counter}/{self.patience if self.patience else '∞'}"
                )
            else:
                patience_counter = 0
                logger.info(f"[HyPER] Score improved: {score_before_iteration:.4f} -> {best_score:.4f}")

            iter_record = {
                "iteration": iteration + 1,
                "best_score": best_score,
                "score_before": score_before_iteration,
                "improved": best_score != score_before_iteration,
                "patience_counter": patience_counter,
                "mini_batch_seed": mini_batch_seed,
                "mini_batch_resample_seed": mini_batch_resample_seed,
                "mini_batch_indices": list(mini_batch_indices),
                "mini_batch_resampled": mini_batch_resampled,
                "candidates": [
                    {
                        "prompt": cand,
                        "mini_batch_score": res.aggregate_score,
                        "n_failed": len(res.failed_examples),
                    }
                    for cand, res in zip(candidates, results)
                ],
                "selected_candidates": [
                    {
                        "prompt": cand_prompt,
                        "mini_batch_score": res.aggregate_score,
                    }
                    for cand_prompt, res in selected
                ],
                "optimized_candidates": optimized_val_scores,
                "recommendations_before_audit": [
                    {
                        "section": r.section,
                        "text": r.text,
                        "weight": r.weight,
                    }
                    for r in recommendations_before_audit
                ],
                "audit_trace": audit_trace,
                "recommendations": [
                    {"section": r.section, "text": r.text} for r in all_recs
                ],
            }
            iteration_history.append(iter_record)

            _bs_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            _pat_str = str(self.patience) if self.patience else "∞"
            logger.info(
                f"[HyPER] Iteration {iteration + 1} done | "
                f"best_score={_bs_str} | "
                f"patience={patience_counter}/{_pat_str}"
            )

            if self.patience and patience_counter >= self.patience:
                logger.info(f"[HyPER] Early stopping triggered: patience {self.patience} reached at iteration {iteration + 1}")
                break

        logger.info(f"\n{'='*60}")
        logger.info(f"[HyPER] Optimization complete!")
        final_score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        logger.info(f"[HyPER] Final best score: {final_score_str}")
        logger.info(f"[HyPER] Total iterations completed: {len(iteration_history)}")
        logger.debug(f"[HyPER] Final best prompt:\n{best_prompt}")
        logger.info(f"{'='*60}")

        return best_prompt, iteration_history


class HyPERMethod(AutoPromptingMethod):
    """HyPER for ``PromptTuner`` / benchmarks (iterative refinement + meta-prompt)."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        n_iterations = kwargs.pop("n_iterations", 5)
        patience = kwargs.pop("patience", None)
        n_candidates = kwargs.pop("n_candidates", 3)
        top_n_candidates = kwargs.pop("top_n_candidates", 3)
        k_samples = kwargs.pop("k_samples", 3)
        mini_batch_size = kwargs.pop("mini_batch_size", 16)
        contrastive_probability = kwargs.pop("contrastive_probability", 0.5)
        contrastive_max_answer_chars = kwargs.pop("contrastive_max_answer_chars", 500)
        feedback_answer_head_chars = kwargs.pop("feedback_answer_head_chars", 500)
        feedback_answer_tail_chars = kwargs.pop("feedback_answer_tail_chars", 500)
        enable_instance_leak_audit = kwargs.pop("enable_instance_leak_audit", True)
        random_seed = kwargs.pop("random_seed", None)
        use_structured_output = kwargs.pop("use_structured_output", False)

        meta_prompt_context = kwargs.pop("meta_prompt_context", None)
        optimizer = HyPEROptimizer(
            model=model,
            evaluator=evaluator,
            n_iterations=n_iterations,
            patience=patience,
            n_candidates=n_candidates,
            top_n_candidates=top_n_candidates,
            k_samples=k_samples,
            mini_batch_size=mini_batch_size,
            contrastive_probability=contrastive_probability,
            contrastive_max_answer_chars=contrastive_max_answer_chars,
            feedback_answer_head_chars=feedback_answer_head_chars,
            feedback_answer_tail_chars=feedback_answer_tail_chars,
            enable_instance_leak_audit=enable_instance_leak_audit,
            random_seed=random_seed,
            use_structured_output=use_structured_output,
        )

        meta_info = meta_prompt_context.copy() if meta_prompt_context else {}
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description

        final_prompt, _ = optimizer.optimize(
            prompt=initial_prompt,
            dataset_split=dataset_split,
            meta_info=meta_info if meta_info else None,
        )
        return final_prompt

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        meta = dict(ctx.config.get("meta_info", {}))
        if "task_description" not in meta:
            td = ctx.config.get("problem_description")
            if td is not None:
                meta["task_description"] = td
        mc = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=ctx.config.get("problem_description"),
            meta_prompt_context=meta if meta else None,
            n_iterations=mc.get("n_iterations", 5),
            patience=mc.get("patience", None),
            n_candidates=mc.get("n_candidates", 3),
            top_n_candidates=mc.get("top_n_candidates", 3),
            k_samples=mc.get("k_samples", 3),
            mini_batch_size=mc.get("mini_batch_size", 16),
            contrastive_probability=mc.get("contrastive_probability", 0.5),
            contrastive_max_answer_chars=mc.get("contrastive_max_answer_chars", 500),
            feedback_answer_head_chars=mc.get("feedback_answer_head_chars", 500),
            feedback_answer_tail_chars=mc.get("feedback_answer_tail_chars", 500),
            enable_instance_leak_audit=mc.get("enable_instance_leak_audit", True),
            random_seed=mc.get("random_seed", None),
            use_structured_output=mc.get("use_structured_output", False),
        )

    def is_data_driven(self):
        return True

    @property
    @override
    def name(self):
        return "hyper"
