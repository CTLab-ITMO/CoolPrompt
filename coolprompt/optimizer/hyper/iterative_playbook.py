"""Iterative HyPER Light method with prompt and playbook refinement."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Optional, Sequence, Tuple, override

from coolprompt.evaluator.evaluator import EvalResultDetailed
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
    TelemetryCallback,
)
from coolprompt.optimizer.hyper.meta_prompt import MetaPromptOptimizer
from coolprompt.optimizer.hyper.playbook import PLAYBOOK_GENERATION_PROMPT
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted

logger = logging.getLogger(__name__)


PLAYBOOK_UPDATE_PROMPT = """You are an expert playbook editor.

Update a reusable playbook for the task using the latest prompt evaluation.
The evaluation contains incorrect or low-scoring answers, their scores, and
reference answers. Extract generalizable failure patterns and turn them into
improved strategies, decision rules, checks, and output requirements.

Do not copy task-instance details into the playbook. Do not solve the individual
examples. Preserve useful existing guidance and change only what the evidence
supports.

Return ONLY one valid JSON object with this structure:
{{
  "task_summary": "short summary of the task",
  "strategies": [
    {{
      "name": "strategy name",
      "when_to_use": "when this strategy applies",
      "steps": ["concrete general step"],
      "checks": ["general quality check"]
    }}
  ],
  "decision_rules": ["general rule for choosing or applying a strategy"],
  "common_failure_modes": ["general failure mode and prevention"],
  "output_contract": ["general requirement for the final answer"]
}}

<task_description>
{task_description}
</task_description>

<current_prompt>
{current_prompt}
</current_prompt>

<current_playbook>
{current_playbook}
</current_playbook>

<evaluation>
Current score: {current_score}
Previous score: {previous_score}
Low-scoring examples:
{failures}
</evaluation>
"""


class HyPERLightPlaybookIterativeMethod(AutoPromptingMethod):
    """Iteratively optimize a prompt while refreshing its task playbook.

    Each iteration performs two meta-level LLM calls:

    1. optimize the current prompt using the current playbook;
    2. update the playbook from low-scoring evaluated examples.

    The evaluator supplies the feedback data. Validation is used for model
    selection when available; otherwise the train split is used. The current
    prompt is carried into the next iteration, while the best-scoring prompt is
    returned at the end.
    """

    def __init__(self) -> None:
        self.last_playbook: Optional[dict[str, Any]] = None
        self.last_iteration_history: list[dict[str, Any]] = []

    def _generate_playbook(self, model: Any, initial_prompt: str) -> dict[str, Any]:
        """Generate the initial structured playbook from the starting prompt."""
        request = PLAYBOOK_GENERATION_PROMPT.format(initial_prompt=initial_prompt)
        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_playbook": raw_result}

    @staticmethod
    def _truncate(text: Any, max_chars: int) -> str:
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "...[truncated]"

    def _format_failures(
        self,
        result: EvalResultDetailed,
        max_failures: int,
        max_answer_chars: int,
    ) -> str:
        """Serialize low-scoring evaluation pairs for playbook refinement."""
        failures = []
        for failure in (result.failed_examples or [])[:max_failures]:
            failures.append(
                {
                    "instance": self._truncate(failure.instance, max_answer_chars),
                    "incorrect_answer": self._truncate(
                        failure.assistant_answer, max_answer_chars
                    ),
                    "parsed_answer": self._truncate(
                        failure.model_answer_parsed, max_answer_chars
                    ),
                    "score": failure.metric_value,
                    "reference_answer": self._truncate(
                        failure.ground_truth, max_answer_chars
                    ),
                }
            )
        return json.dumps(failures, ensure_ascii=False, indent=2)

    @staticmethod
    def _sample_train_batch(
        samples: Sequence[str],
        targets: Sequence[str | int],
        *,
        batch_size: int,
        pool_size: int,
        seed: int,
    ) -> tuple[list[str], list[str | int], list[int]]:
        """Sample a deterministic batch from the first ``pool_size`` train rows."""
        pool_length = min(len(samples), pool_size)
        batch_length = min(batch_size, pool_length)
        rng = random.Random(seed)
        indices = rng.sample(range(pool_length), batch_length)
        return (
            [samples[index] for index in indices],
            [targets[index] for index in indices],
            indices,
        )

    def _update_playbook(
        self,
        model: Any,
        *,
        task_description: str,
        current_prompt: str,
        current_playbook: dict[str, Any],
        current_score: float,
        previous_score: float,
        result: EvalResultDetailed,
        max_failures: int,
        max_answer_chars: int,
        update_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Update the playbook using evaluation failures and metric values."""
        template = update_prompt or PLAYBOOK_UPDATE_PROMPT
        request = template.format(
            task_description=task_description,
            current_prompt=current_prompt,
            current_playbook=json.dumps(
                current_playbook, ensure_ascii=False, indent=2
            ),
            current_score=current_score,
            previous_score=previous_score,
            failures=self._format_failures(
                result, max_failures=max_failures, max_answer_chars=max_answer_chars
            ),
        )
        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("Playbook update did not return JSON; keeping current playbook")
        return current_playbook

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ) -> str:
        """Run iterative prompt/playbook optimization."""
        if dataset_split is None or evaluator is None:
            raise ValueError(
                "hyper_light_playbook_iterative requires dataset_split and evaluator"
            )

        train_samples, val_samples, train_targets, val_targets = dataset_split
        selection_samples: Sequence[str] = val_samples or train_samples
        selection_targets: Sequence[str | int] = val_targets or train_targets
        if not selection_samples:
            raise ValueError(
                "hyper_light_playbook_iterative requires non-empty train or validation data"
            )

        telemetry_callback: Optional[TelemetryCallback] = kwargs.pop(
            "telemetry_callback", None
        )
        hyper_meta_info = kwargs.pop("hyper_meta_info", None)
        hyper_meta_prompt = kwargs.pop("hyper_meta_prompt", None)
        use_structured_output = kwargs.pop("use_structured_output", False)
        n_iterations = kwargs.pop(
            "n_iterations", kwargs.pop("num_epochs", kwargs.pop("epochs", 5))
        )
        train_batch_size = kwargs.pop("train_batch_size", 50)
        train_pool_size = kwargs.pop("train_pool_size", 300)
        random_seed = kwargs.pop("random_seed", 42)
        max_failures = kwargs.pop("k_samples", kwargs.pop("max_failures", 3))
        max_answer_chars = kwargs.pop("max_answer_chars", 2000)
        playbook_update_prompt = kwargs.pop("playbook_update_prompt", None)
        playbook_prompt = kwargs.pop("playbook_prompt", None)

        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        if train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if train_pool_size < 1:
            raise ValueError("train_pool_size must be >= 1")

        if playbook_prompt is None:
            playbook = self._generate_playbook(model, initial_prompt)
        else:
            raw_playbook = get_model_answer_extracted(
                model, playbook_prompt.format(initial_prompt=initial_prompt)
            )
            parsed_playbook = extract_json(raw_playbook)
            playbook = (
                parsed_playbook
                if isinstance(parsed_playbook, dict)
                else {"raw_playbook": raw_playbook}
            )

        meta_info = hyper_meta_info.copy() if hyper_meta_info else {}
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description

        optimizer_kwargs = {
            "model": model,
            "use_structured_output": use_structured_output,
            **kwargs,
        }
        if hyper_meta_prompt is not None:
            optimizer_kwargs["meta_prompt"] = hyper_meta_prompt
        optimizer = MetaPromptOptimizer(**optimizer_kwargs)

        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score: Optional[float] = None
        previous_score: Optional[float] = None
        history: list[dict[str, Any]] = []

        for iteration in range(1, n_iterations + 1):
            train_batch, train_batch_targets, train_batch_indices = (
                self._sample_train_batch(
                    train_samples,
                    train_targets,
                    batch_size=train_batch_size,
                    pool_size=train_pool_size,
                    seed=random_seed + iteration - 1,
                )
            )
            if not train_batch:
                raise ValueError("The train pool produced an empty batch")

            iteration_meta_info = meta_info.copy()
            iteration_meta_info["playbook"] = playbook
            iteration_meta_info["iteration"] = iteration

            optimized_prompt = optimizer.optimize(
                prompt=current_prompt,
                meta_info=iteration_meta_info,
                n_prompts=1,
            )
            train_evaluation = evaluator.evaluate(
                optimized_prompt,
                train_batch,
                train_batch_targets,
                failed_examples=max_failures,
                return_detailed=True,
            )
            train_score = train_evaluation.aggregate_score

            validation_score: Optional[float] = None
            if val_samples:
                validation_score = evaluator.evaluate(
                    optimized_prompt,
                    list(selection_samples),
                    list(selection_targets),
                )
            current_score = validation_score if validation_score is not None else train_score

            logger.info(
                "[IterativePlaybook] iteration=%d/%d train_score=%.6f "
                "train_batch_size=%d train_pool_size=%d seed=%d",
                iteration,
                n_iterations,
                train_score,
                len(train_batch),
                min(len(train_samples), train_pool_size),
                random_seed + iteration - 1,
            )
            if validation_score is not None:
                logger.info(
                    "[IterativePlaybook] iteration=%d/%d validation_score=%.6f",
                    iteration,
                    n_iterations,
                    validation_score,
                )

            if best_score is None or current_score > best_score:
                best_prompt = optimized_prompt
                best_score = current_score

            updated_playbook = self._update_playbook(
                model,
                task_description=problem_description or initial_prompt,
                current_prompt=optimized_prompt,
                current_playbook=playbook,
                current_score=train_score,
                previous_score=previous_score if previous_score is not None else train_score,
                result=train_evaluation,
                max_failures=max_failures,
                max_answer_chars=max_answer_chars,
                update_prompt=playbook_update_prompt,
            )

            logger.info(
                "[IterativePlaybook] iteration=%d/%d optimized_prompt=%s\n"
                "[IterativePlaybook] iteration=%d/%d playbook=%s",
                iteration,
                n_iterations,
                optimized_prompt,
                iteration,
                n_iterations,
                json.dumps(updated_playbook, ensure_ascii=False, indent=2),
            )

            history.append(
                {
                    "iteration": iteration,
                    "prompt": optimized_prompt,
                    "score": current_score,
                    "train_score": train_score,
                    "validation_score": validation_score,
                    "previous_score": previous_score,
                    "best_score": best_score,
                    "train_batch_indices": train_batch_indices,
                    "train_batch_size": len(train_batch),
                    "train_pool_size": min(len(train_samples), train_pool_size),
                    "random_seed": random_seed + iteration - 1,
                    "n_failures": len(train_evaluation.failed_examples or []),
                    "playbook": updated_playbook,
                }
            )
            current_prompt = optimized_prompt
            previous_score = train_score
            playbook = updated_playbook

            if telemetry_callback is not None:
                telemetry_callback(
                    iteration=iteration,
                    best_score=best_score,
                    best_prompt=best_prompt,
                )

        self.last_playbook = playbook
        self.last_iteration_history = history
        return best_prompt

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run the iterative method from a benchmark configuration."""
        meta = dict(ctx.config.get("meta_info", {}))
        method_config = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=ctx.config.get("problem_description"),
            hyper_meta_info=meta if meta else None,
            hyper_meta_prompt=method_config.get("hyper_meta_prompt"),
            playbook_prompt=method_config.get("playbook_prompt"),
            playbook_update_prompt=method_config.get("playbook_update_prompt"),
            use_structured_output=method_config.get("use_structured_output", False),
            n_iterations=method_config.get(
                "n_iterations", method_config.get("num_epochs", 5)
            ),
            train_batch_size=method_config.get("train_batch_size", 50),
            train_pool_size=method_config.get("train_pool_size", 300),
            random_seed=method_config.get("random_seed", 42),
            max_failures=method_config.get("max_failures", 3),
            max_answer_chars=method_config.get("max_answer_chars", 2000),
        )

    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "hyper_light_playbook_iterative"
