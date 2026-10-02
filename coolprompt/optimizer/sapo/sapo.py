"""Segment-based Automatic Prompt Optimization (SAPO) for CoolPrompt."""

from __future__ import annotations

import time
from typing import Any, override

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
    TelemetryCallback,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json

from .prompt_templates import (
    CANDIDATE_GENERATION_TEMPLATE,
    SEGMENTATION_TEMPLATE,
    WEAKNESS_ANALYSIS_TEMPLATE,
)

_SEGMENTS = ("role", "context", "tasks", "output_format")


class PromptSegments(BaseModel):
    """A prompt split into the four segments used by SAPO."""

    role: str = ""
    context: str = ""
    tasks: str = ""
    output_format: str = ""


class WeaknessAnalysis(BaseModel):
    """Segment-level diagnosis produced from contrastive examples."""

    weak_segments: list[str] = Field(default_factory=list)
    strong_segments: list[str] = Field(default_factory=list)
    recommendations: dict[str, str] = Field(default_factory=dict)


class CandidatePrompts(BaseModel):
    """Structured response for candidate prompt generation."""

    prompts: list[str] = Field(default_factory=list)


class SAPOOptimizer:
    """Optimize a prompt by diagnosing and rewriting its logical segments."""

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        *,
        n_iterations: int = 5,
        n_candidates: int = 5,
        early_stopping_rounds: int = 3,
        examples_per_side: int = 5,
        max_request_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> None:
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative")
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be at least 1")
        self.model = model
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.early_stopping_rounds = early_stopping_rounds
        self.examples_per_side = max(1, examples_per_side)
        self.max_request_retries = max(1, max_request_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.telemetry_callback = telemetry_callback
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    def _retry(self, operation: str, function):
        last_error: Exception | None = None
        for attempt in range(1, self.max_request_retries + 1):
            try:
                return function()
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                last_error = exc
                if attempt < self.max_request_retries:
                    logger.warning(
                        "SAPO %s failed (%s/%s): %s",
                        operation,
                        attempt,
                        self.max_request_retries,
                        exc,
                    )
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"SAPO {operation} failed after {self.max_request_retries} attempts"
        ) from last_error

    def _invoke_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Use native structured output when available, with JSON fallback."""

        def invoke() -> BaseModel:
            if hasattr(self.model, "with_structured_output"):
                try:
                    result = self.model.with_structured_output(schema).invoke(prompt)
                    if isinstance(result, schema):
                        return result
                    return schema.model_validate(result)
                except (AttributeError, NotImplementedError, TypeError, ValueError):
                    pass
            parsed = extract_json(self._response_text(self.model.invoke(prompt)))
            if parsed is None:
                raise ValueError(f"Model did not return JSON for {schema.__name__}")
            return schema.model_validate(parsed)

        return self._retry(schema.__name__, invoke)

    def _evaluate(
        self, prompt: str, dataset: list[str], targets: list[Any]
    ) -> tuple[float, list[float], list[str]]:
        result = self.evaluator.evaluate(
            prompt=prompt,
            dataset=dataset,
            targets=targets,
            return_detailed=True,
        )
        return (
            float(result.aggregate_score),
            [float(score) for score in result.score_per_task],
            list(result.raw_outputs),
        )

    def _extract_segments(self, prompt: str) -> PromptSegments:
        return self._invoke_structured(
            SEGMENTATION_TEMPLATE.format(prompt=prompt), PromptSegments
        )

    def _ranked_examples(
        self,
        dataset: list[str],
        targets: list[Any],
        scores: list[float],
        outputs: list[str],
    ) -> tuple[str, str]:
        indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        count = min(self.examples_per_side, len(indices))

        def render(selected: list[int]) -> str:
            return "\n\n".join(
                f"Input: {dataset[index]}\nReference: {targets[index]}\n"
                f"Model response: {outputs[index]}\nScore: {scores[index]:.4f}"
                for index in selected
            )

        return render(indices[:count]), render(indices[-count:])

    @staticmethod
    def _sanitize_analysis(analysis: WeaknessAnalysis) -> WeaknessAnalysis:
        weak = list(dict.fromkeys(x for x in analysis.weak_segments if x in _SEGMENTS))
        strong = list(
            dict.fromkeys(
                x for x in analysis.strong_segments if x in _SEGMENTS and x not in weak
            )
        )
        recommendations = {
            key: str(value).strip()
            for key, value in analysis.recommendations.items()
            if key in weak and str(value).strip()
        }
        if not weak:
            weak = ["tasks"]
        for segment in weak:
            recommendations.setdefault(
                segment, f"Make the {segment} clearer and more specific."
            )
        return WeaknessAnalysis(
            weak_segments=weak,
            strong_segments=strong,
            recommendations=recommendations,
        )

    def _analyze(
        self,
        prompt: str,
        segments: PromptSegments,
        dataset: list[str],
        targets: list[Any],
        scores: list[float],
        outputs: list[str],
    ) -> WeaknessAnalysis:
        best, worst = self._ranked_examples(dataset, targets, scores, outputs)
        analysis = self._invoke_structured(
            WEAKNESS_ANALYSIS_TEMPLATE.format(
                prompt=prompt,
                best_examples=best,
                worst_examples=worst,
                **segments.model_dump(),
            ),
            WeaknessAnalysis,
        )
        return self._sanitize_analysis(analysis)

    def _generate_candidates(
        self,
        prompt: str,
        segments: PromptSegments,
        analysis: WeaknessAnalysis,
    ) -> list[str]:
        recommendations = "\n".join(
            f"- {key}: {value}" for key, value in analysis.recommendations.items()
        )
        result = self._invoke_structured(
            CANDIDATE_GENERATION_TEMPLATE.format(
                n_candidates=self.n_candidates,
                current_prompt=prompt,
                weak_segments=", ".join(analysis.weak_segments),
                strong_segments=", ".join(analysis.strong_segments) or "none",
                recommendations=recommendations,
                **segments.model_dump(),
            ),
            CandidatePrompts,
        )
        candidates: list[str] = []
        seen = {prompt.strip().casefold()}
        for candidate in result.prompts:
            candidate = candidate.strip()
            key = candidate.casefold()
            if candidate and key not in seen:
                candidates.append(candidate)
                seen.add(key)
        if not candidates:
            logger.warning("SAPO produced no distinct candidates; stopping")
        return candidates[: self.n_candidates]

    def optimize(
        self,
        initial_prompt: str,
        dataset_split: tuple[list[str], list[str], list[Any], list[Any]],
    ) -> str:
        train_data, val_data, train_targets, val_targets = map(list, dataset_split)
        if not train_data or not train_targets:
            raise ValueError("SAPO requires a non-empty training dataset and targets")
        if not val_data:
            val_data, val_targets = train_data, train_targets

        current_prompt = initial_prompt.strip()
        train_score, train_scores, train_outputs = self._evaluate(
            current_prompt, train_data, train_targets
        )
        best_score, _, _ = self._evaluate(current_prompt, val_data, val_targets)
        best_prompt = current_prompt
        self.history = [
            {
                "iteration": 0,
                "prompt": current_prompt,
                "train_score": train_score,
                "val_score": best_score,
                "improved": False,
            }
        ]
        if self.telemetry_callback:
            self.telemetry_callback(0, best_score, best_prompt)

        no_improvement = 0
        for iteration in range(1, self.n_iterations + 1):
            segments = self._extract_segments(current_prompt)
            analysis = self._analyze(
                current_prompt,
                segments,
                train_data,
                train_targets,
                train_scores,
                train_outputs,
            )
            candidates = self._generate_candidates(current_prompt, segments, analysis)
            if not candidates:
                break

            evaluated = []
            for candidate in candidates:
                candidate_train = self._evaluate(candidate, train_data, train_targets)
                candidate_val = self._evaluate(candidate, val_data, val_targets)[0]
                evaluated.append((candidate_val, candidate, candidate_train))
            candidate_score, candidate_prompt, candidate_train = max(
                evaluated, key=lambda item: item[0]
            )
            improved = candidate_score > best_score
            if improved:
                best_score = candidate_score
                best_prompt = candidate_prompt
                current_prompt = candidate_prompt
                train_score, train_scores, train_outputs = candidate_train
                no_improvement = 0
            else:
                no_improvement += 1

            self.history.append(
                {
                    "iteration": iteration,
                    "prompt": current_prompt,
                    "train_score": train_score,
                    "val_score": best_score,
                    "segments": segments.model_dump(),
                    "analysis": analysis.model_dump(),
                    "candidates": candidates,
                    "candidate_scores": [item[0] for item in evaluated],
                    "improved": improved,
                }
            )
            logger.info(
                "SAPO iteration %s: best validation score %.4f%s",
                iteration,
                best_score,
                " (improved)" if improved else "",
            )
            if self.telemetry_callback:
                self.telemetry_callback(iteration, best_score, best_prompt)
            if no_improvement >= self.early_stopping_rounds:
                break
        return best_prompt


class SAPOMethod(AutoPromptingMethod):
    """CoolPrompt adapter for Segment-based Automatic Prompt Optimization."""

    def __init__(self) -> None:
        self.last_optimizer: SAPOOptimizer | None = None

    @override
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ) -> str:
        del problem_description
        telemetry_callback = kwargs.pop("telemetry_callback", None)
        self.last_optimizer = SAPOOptimizer(
            model=model,
            evaluator=evaluator,
            n_iterations=kwargs.pop("n_iterations", 5),
            n_candidates=kwargs.pop("n_candidates", 5),
            early_stopping_rounds=kwargs.pop(
                "early_stopping_rounds", kwargs.pop("patience", 3)
            ),
            examples_per_side=kwargs.pop("examples_per_side", 5),
            max_request_retries=kwargs.pop("max_request_retries", 3),
            retry_delay_seconds=kwargs.pop("retry_delay_seconds", 1.0),
            telemetry_callback=telemetry_callback,
        )
        if kwargs:
            logger.debug("Ignoring unsupported SAPO options: %s", sorted(kwargs))
        return self.last_optimizer.optimize(initial_prompt, dataset_split)

    @override
    def run_configured_benchmark(self, ctx: BenchmarkContext, start_prompt: str) -> str:
        config = dict(ctx.config.get("method", {}))
        return self.optimize(
            model=ctx.model,
            initial_prompt=start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=ctx.config.get("problem_description"),
            **config,
        )

    @override
    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "sapo"
