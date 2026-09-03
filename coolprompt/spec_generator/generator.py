"""High-level orchestration for synthetic-data generation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field

from coolprompt.data_generator.pydantic_formatters import (
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskStructuredOutputSchema,
)
from coolprompt.spec_generator.distribution import (
    GenerationState,
    TaggedGenerationBatch,
    TaskDistribution,
    _TaskDistributionBuilder,
    build_generation_targets,
    validate_axis_tags,
)
from coolprompt.spec_generator.models import (
    Example,
    GenerationContext,
    GenerationResult,
    TaskSpecDraft,
)
from coolprompt.spec_generator.prompt_builder import GenerationPromptBuilder
from coolprompt.spec_generator.spec_builder import SpecBuilder
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.spec_generator.validation.format import Deduplicator, ExampleValidator
from coolprompt.spec_generator.validation.judge import LLMJudge
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_json

_OUTPUT_SCHEMAS: dict[Task, type[BaseModel]] = {
    Task.CLASSIFICATION: ClassificationTaskStructuredOutputSchema,
    Task.GENERATION: GenerationTaskStructuredOutputSchema,
}


class GenerationResponseError(ValueError):
    """Raised when a generation response cannot be used safely."""


class MultiReferenceGeneratedExample(BaseModel):
    """Structured output for non-tagged multi-reference generation."""

    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    references: list[str] = Field(default_factory=list)


class MultiReferenceGenerationBatch(BaseModel):
    examples: list[MultiReferenceGeneratedExample]


def _split_count(total: int, corner_ratio: float) -> tuple[int, int]:
    corner = int(total * corner_ratio)
    return total - corner, corner


def _batch_sizes(total: int, batch_size: int) -> Iterator[int]:
    remaining = total
    while remaining > 0:
        current = min(remaining, batch_size)
        yield current
        remaining -= current


def _validate_generation_args(
        num_samples: int,
        batch_size: int,
        corner_ratio: float,
        candidate_multiplier: float,
        valid_outputs_per_example: int,
) -> None:
    if not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 100")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= corner_ratio <= 1.0:
        raise ValueError("corner_ratio must be between 0.0 and 1.0")
    if not 1.0 <= candidate_multiplier <= 3.0:
        raise ValueError("candidate_multiplier must be between 1.0 and 3.0")
    if not 1 <= valid_outputs_per_example <= 5:
        raise ValueError("valid_outputs_per_example must be between 1 and 5")


def _extract_examples(payload: Any) -> list[Any]:
    if isinstance(payload, AIMessage):
        payload = payload.content
    if isinstance(payload, str):
        payload = extract_json(payload)

    if isinstance(payload, BaseModel):
        examples = getattr(payload, "examples", None)
    elif isinstance(payload, dict):
        examples = payload.get("examples")
    else:
        examples = None

    if not isinstance(examples, list):
        raise GenerationResponseError("Generation response does not contain an examples list.")
    if not examples:
        raise GenerationResponseError("Generation response contains no examples.")
    return examples


class SyntheticDataGenerator:
    """Generate synthetic examples from an immutable generation context."""

    def __init__(
            self,
            model: BaseLanguageModel,
            detector_confidence_threshold: float = 0.7,
            retry_config: RetryConfig | None = None,
            max_topup_attempts: int = 10,
            judge_quality_threshold: float = 0.7,
            judge_batch_size: int = 15,
            *,
            task_spec_model: BaseLanguageModel | None = None,
            judge_model: BaseLanguageModel | None = None,
    ) -> None:
        self._model = model
        self._judge_model = judge_model or model
        self._retry_config = retry_config or RetryConfig()
        self._max_topup_attempts = max_topup_attempts
        self._judge_quality_threshold = judge_quality_threshold
        self._judge_batch_size = judge_batch_size
        self._spec_builder = SpecBuilder(
            model=model,
            detector_confidence_threshold=detector_confidence_threshold,
            retry_config=self._retry_config,
            task_spec_model=task_spec_model,
        )
        self._prompt_builder = GenerationPromptBuilder()
        self._distribution_builder = _TaskDistributionBuilder(
            model=model,
            retry_config=self._retry_config,
        )
        self._last_distribution: TaskDistribution | None = None
        self._last_generation_state: GenerationState | None = None

    def build_context(
            self,
            prompt: str,
            *,
            draft: TaskSpecDraft | None = None,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            detect_dataset: bool = False,
            dataset_name: str | None = None,
    ) -> GenerationContext:
        return self._spec_builder.build(
            prompt=prompt,
            examples=examples,
            draft=draft,
            detect_dataset=detect_dataset,
            dataset_name=dataset_name,
        )

    def generate(
            self,
            prompt: str,
            *,
            draft: TaskSpecDraft | None = None,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            distribution_examples: Sequence[tuple[str, str] | Example] | None = None,
            detect_dataset: bool = False,
            num_samples: int = 8,
            batch_size: int = 15,
            corner_ratio: float = 0.4,
            structural_validation: bool = True,
            judge_regular: bool = False,
            judge_corner_cases: bool = False,
            use_task_distribution: bool = False,
            feedback_controlled: bool = False,
            corner_phase: bool = False,
            candidate_multiplier: float = 1.5,
            valid_outputs_per_example: int = 1,
    ) -> GenerationResult:
        """Generate exactly ``num_samples`` synthetic examples.

        Baseline behavior is unchanged when ``use_task_distribution`` and
        ``feedback_controlled`` are false. Feedback mode additionally enables semantic
        and structural novelty filtering and generates a candidate pool larger than the
        number of examples that must be accepted.
        """

        _validate_generation_args(
            num_samples,
            batch_size,
            corner_ratio,
            candidate_multiplier,
            valid_outputs_per_example,
        )

        context = self.build_context(
            prompt,
            draft=draft,
            examples=examples,
            detect_dataset=detect_dataset,
        )
        self._validate_context(context)

        if context.spec.task is Task.CLASSIFICATION and valid_outputs_per_example != 1:
            raise ValueError("Multi-reference generation is only supported for generation tasks.")

        if feedback_controlled and not use_task_distribution:
            raise ValueError("feedback_controlled requires use_task_distribution=True")
        if corner_phase and not feedback_controlled:
            raise ValueError("corner_phase requires feedback_controlled=True")
        if not structural_validation and (judge_regular or judge_corner_cases):
            raise ValueError("LLM judging requires structural_validation=True")

        distribution_reference = tuple(
            item if isinstance(item, Example) else Example(input=item[0], output=item[1])
            for item in (distribution_examples or ())
        )

        effective_reference = distribution_reference or context.seed_examples

        distribution = (
            self._distribution_builder.build(
                prompt=prompt,
                spec=context.spec,
                examples=context.seed_examples,
                reference_examples=effective_reference,
            )
            if use_task_distribution
            else None
        )

        self._last_distribution = distribution
        self._last_generation_state = None

        if feedback_controlled:
            assert distribution is not None
            generated = self._generate_feedback_controlled(
                context=context,
                distribution=distribution,
                num_samples=num_samples,
                batch_size=batch_size,
                judge_regular=judge_regular,
                judge_corner_cases=judge_corner_cases,
                corner_phase=corner_phase,
                corner_ratio=corner_ratio,
                candidate_multiplier=candidate_multiplier,
                reference_examples=effective_reference,
                valid_outputs_per_example=valid_outputs_per_example,
                structural_validation=structural_validation,
            )
        else:
            regular_count, corner_count = _split_count(num_samples, corner_ratio)

            if not context.spec.corner_cases:
                regular_count = num_samples
                corner_count = 0

            if structural_validation:
                generated = self._generate_validated(
                    context=context,
                    regular_count=regular_count,
                    corner_count=corner_count,
                    batch_size=batch_size,
                    judge_regular=judge_regular,
                    judge_corner_cases=judge_corner_cases,
                    distribution=distribution,
                    valid_outputs_per_example=valid_outputs_per_example,
                )
            else:
                generated = self._generate_unvalidated(
                    context=context,
                    regular_count=regular_count,
                    corner_count=corner_count,
                    batch_size=batch_size,
                    distribution=distribution,
                    valid_outputs_per_example=valid_outputs_per_example,
                )

        if len(generated) != num_samples:
            raise RuntimeError(f"Expected {num_samples} examples, received {len(generated)}")

        return GenerationResult(
            examples=tuple(self._coerce_example(item) for item in generated),
            context=context,
        )

    @staticmethod
    def _coerce_example(item: Any) -> Example:
        if isinstance(item, Example):
            return item
        if isinstance(item, BaseModel):
            payload = item.model_dump()
            return Example(
                input=payload["input"],
                output=payload["output"],
                references=tuple(payload.get("references") or ()),
            )
        if isinstance(item, dict):
            return Example(
                input=item["input"],
                output=item["output"],
                references=tuple(item.get("references") or ()),
            )
        return Example(
            input=getattr(item, "input"),
            output=getattr(item, "output"),
            references=tuple(getattr(item, "references", ()) or ()),
        )

    @staticmethod
    def _validate_context(context: GenerationContext) -> None:
        if context.spec.task not in _OUTPUT_SCHEMAS:
            supported = ", ".join(task.value for task in _OUTPUT_SCHEMAS)
            raise ValueError(f"Unsupported task {context.spec.task!r}; supported tasks: {supported}")

    def _generate_validated(
            self,
            context: GenerationContext,
            regular_count: int,
            corner_count: int,
            batch_size: int,
            judge_regular: bool,
            judge_corner_cases: bool,
            distribution: TaskDistribution | None = None,
            valid_outputs_per_example: int = 1,
    ) -> list[Example]:
        pipeline = self._build_pipeline(novelty=False, min_references=max(0, valid_outputs_per_example - 1))

        regular = self._generate_validated_group(
            pipeline,
            context,
            regular_count,
            batch_size,
            is_corner=False,
            apply_judge=judge_regular,
            reset_deduplicator=True,
            distribution=distribution,
            valid_outputs_per_example=valid_outputs_per_example,
        )
        corner = self._generate_validated_group(
            pipeline,
            context,
            corner_count,
            batch_size,
            is_corner=True,
            apply_judge=judge_corner_cases,
            reset_deduplicator=not regular,
            distribution=distribution,
            valid_outputs_per_example=valid_outputs_per_example,
        )
        return regular + corner

    def _generate_validated_group(
            self,
            pipeline: ValidationPipeline,
            context: GenerationContext,
            target: int,
            batch_size: int,
            *,
            is_corner: bool,
            apply_judge: bool,
            reset_deduplicator: bool,
            distribution: TaskDistribution | None = None,
            valid_outputs_per_example: int = 1,
    ) -> list[Example]:
        if target <= 0:
            return []

        result = pipeline.run(
            producer=lambda remaining: self._generate_group(
                context,
                remaining,
                batch_size,
                is_corner=is_corner,
                distribution=distribution,
                valid_outputs_per_example=valid_outputs_per_example,
            ),
            context=context,
            target_n=target,
            judge=apply_judge,
            is_corner=is_corner,
            reset_deduplicator=reset_deduplicator,
        )
        if len(result) < target:
            group = "corner" if is_corner else "regular"
            raise RuntimeError(f"Could not generate enough {group} examples: {len(result)}/{target}")
        return result

    def _generate_unvalidated(
            self,
            context: GenerationContext,
            regular_count: int,
            corner_count: int,
            batch_size: int,
            distribution: TaskDistribution | None = None,
            valid_outputs_per_example: int = 1,
    ) -> list[Any]:
        regular = self._generate_group(
            context,
            regular_count,
            batch_size,
            is_corner=False,
            distribution=distribution,
            valid_outputs_per_example=valid_outputs_per_example,
        )
        corner = self._generate_group(
            context,
            corner_count,
            batch_size,
            is_corner=True,
            distribution=distribution,
            valid_outputs_per_example=valid_outputs_per_example,
        )
        return regular + corner

    def _generate_group(
            self,
            context: GenerationContext,
            total: int,
            batch_size: int,
            *,
            is_corner: bool,
            distribution: TaskDistribution | None = None,
            valid_outputs_per_example: int = 1,
    ) -> list[Any]:
        generated: list[Any] = []
        for size in _batch_sizes(total, batch_size):
            if is_corner:
                cases = context.spec.corner_cases
                selected = random.sample(cases, min(len(cases), size))
                request = self._prompt_builder.corner(
                    context,
                    size,
                    corner_cases=selected,
                    valid_outputs_per_example=valid_outputs_per_example,
                )
            elif distribution is None:
                request = self._prompt_builder.regular(
                    context,
                    size,
                    valid_outputs_per_example=valid_outputs_per_example,
                )
            else:
                request = self._prompt_builder.distribution_aware(
                    context,
                    size,
                    distribution,
                    valid_outputs_per_example=valid_outputs_per_example,
                )

            generated.extend(
                self._call_model(
                    request,
                    context.spec.task,
                    with_axis_tags=distribution is not None and not is_corner,
                    valid_outputs_per_example=valid_outputs_per_example,
                )
            )
        return generated

    def _call_model(
            self,
            request: str,
            task: Task,
            *,
            with_axis_tags: bool = False,
            valid_outputs_per_example: int = 1,
    ) -> list[Any]:
        multi_reference = task is Task.GENERATION and valid_outputs_per_example > 1
        if with_axis_tags:
            schema = TaggedGenerationBatch
        elif multi_reference:
            schema = MultiReferenceGenerationBatch
        else:
            schema = _OUTPUT_SCHEMAS[task]

        chat_model = resolve_chat_model(self._model)

        def invoke() -> list[Any]:
            if chat_model is None:
                output = self._model.invoke(request)
            else:
                method = "function_calling" if (with_axis_tags or multi_reference) else "json_schema"
                output = chat_model.with_structured_output(
                    schema=schema,
                    method=method,
                ).invoke(request)
            return _extract_examples(output)

        return invoke_with_retry(
            invoke,
            self._retry_config,
            extra_retry_exceptions=(GenerationResponseError,),
        )

    def _generate_feedback_controlled(
            self,
            *,
            context: GenerationContext,
            distribution: TaskDistribution,
            num_samples: int,
            batch_size: int,
            judge_regular: bool,
            judge_corner_cases: bool,
            corner_phase: bool,
            corner_ratio: float,
            candidate_multiplier: float,
            reference_examples: Sequence[Example],
            valid_outputs_per_example: int,
            structural_validation: bool,
    ) -> list[Example]:
        """Generate, observe accepted coverage/novelty, then target the next batch."""

        pipeline = self._build_pipeline(
            novelty=structural_validation,
            min_references=max(0, valid_outputs_per_example - 1),
        )
        state = GenerationState()
        accepted: list[Example] = []

        corner_cases = tuple(context.spec.corner_cases) if corner_phase else ()

        corner_budget = (
            int(num_samples * corner_ratio)
            if corner_phase and corner_cases
            else 0
        )

        regular_budget = num_samples - corner_budget

        if regular_budget:
            first_n = min(batch_size, regular_budget)
            batch, tags = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=first_n,
                batch_size=batch_size,
                candidate_multiplier=candidate_multiplier,
                apply_judge=judge_regular,
                reset_deduplicator=True,
                targets=None,
                avoid=(),
                accepted_examples=accepted,
                reference_examples=reference_examples,
                valid_outputs_per_example=valid_outputs_per_example,
            )
            accepted.extend(batch)
            self._record_feedback_batch(state, distribution, context, batch, tags)

        while len(accepted) < regular_budget:
            remaining = regular_budget - len(accepted)
            current_n = min(batch_size, remaining)
            targets, avoid = build_generation_targets(
                distribution,
                state,
                batch_size=current_n,
                remaining_budget=remaining,
                total_target=regular_budget,
            )
            batch, tags = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=current_n,
                batch_size=batch_size,
                candidate_multiplier=candidate_multiplier,
                apply_judge=judge_regular,
                reset_deduplicator=False,
                targets=targets,
                avoid=avoid,
                accepted_examples=accepted,
                reference_examples=reference_examples,
                valid_outputs_per_example=valid_outputs_per_example,
            )
            if not batch:
                break
            accepted.extend(batch)
            self._record_feedback_batch(state, distribution, context, batch, tags)

        if corner_budget:
            corner, corner_tags = self._run_corner_phase(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                corner_cases=corner_cases,
                target_n=corner_budget,
                apply_judge=judge_corner_cases,
                reset_deduplicator=not accepted,
                accepted_examples=accepted,
                candidate_multiplier=candidate_multiplier,
                reference_examples=reference_examples,
                valid_outputs_per_example=valid_outputs_per_example,
            )
            accepted.extend(corner)
            self._record_feedback_batch(state, distribution, context, corner, corner_tags)

        while len(accepted) < num_samples:
            missing = num_samples - len(accepted)
            batch, tags = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=min(batch_size, missing),
                batch_size=batch_size,
                candidate_multiplier=candidate_multiplier,
                apply_judge=judge_regular,
                reset_deduplicator=not accepted,
                targets=None,
                avoid=(),
                accepted_examples=accepted,
                reference_examples=reference_examples,
                valid_outputs_per_example=valid_outputs_per_example,
            )
            if not batch:
                break
            accepted.extend(batch)
            self._record_feedback_batch(state, distribution, context, batch, tags)

        if len(accepted) < num_samples:
            raise RuntimeError(
                f"Could not generate enough feedback-controlled examples: {len(accepted)}/{num_samples}"
            )
        self._last_generation_state = state
        return accepted[:num_samples]

    def _run_feedback_batch(
            self,
            *,
            pipeline: ValidationPipeline,
            context: GenerationContext,
            distribution: TaskDistribution,
            target_n: int,
            batch_size: int,
            candidate_multiplier: float,
            apply_judge: bool,
            reset_deduplicator: bool,
            targets: Sequence[dict[str, Any]] | None,
            avoid: Sequence[dict[str, Any]],
            accepted_examples: Sequence[Example],
            reference_examples: Sequence[Example],
            valid_outputs_per_example: int,
    ) -> tuple[list[Example], dict[tuple[str, str], dict[str, str]]]:
        tag_cache: dict[tuple[str, str], dict[str, str]] = {}

        def producer(remaining: int) -> list[Any]:
            candidate_n = min(
                math.ceil(batch_size * candidate_multiplier),
                max(remaining, math.ceil(remaining * candidate_multiplier)),
            )
            if targets is None:
                request = self._prompt_builder.distribution_aware(
                    context,
                    candidate_n,
                    distribution,
                    accepted_examples=accepted_examples,
                    reference_examples=reference_examples,
                    valid_outputs_per_example=valid_outputs_per_example,
                )
            else:
                request = self._prompt_builder.targeted(
                    context,
                    candidate_n,
                    distribution,
                    targets=targets,
                    avoid=avoid,
                    accepted_examples=accepted_examples,
                    reference_examples=reference_examples,
                    valid_outputs_per_example=valid_outputs_per_example,
                )

            raw = self._call_model(
                request,
                context.spec.task,
                with_axis_tags=True,
                valid_outputs_per_example=valid_outputs_per_example,
            )
            self._cache_axis_tags(tag_cache, raw)
            return raw

        result = pipeline.run(
            producer=producer,
            context=context,
            target_n=target_n,
            judge=apply_judge,
            is_corner=False,
            reset_deduplicator=reset_deduplicator,
        )
        return result, tag_cache

    def _run_corner_phase(
            self,
            *,
            pipeline: ValidationPipeline,
            context: GenerationContext,
            distribution: TaskDistribution,
            corner_cases: Sequence[str],
            target_n: int,
            apply_judge: bool,
            reset_deduplicator: bool,
            accepted_examples: Sequence[Example],
            candidate_multiplier: float,
            reference_examples: Sequence[Example],
            valid_outputs_per_example: int,
    ) -> tuple[list[Example], dict[tuple[str, str], dict[str, str]]]:
        tag_cache: dict[tuple[str, str], dict[str, str]] = {}

        def producer(remaining: int) -> list[Any]:
            requested = max(remaining, math.ceil(remaining * candidate_multiplier))

            selected = tuple(
                corner_cases[index % len(corner_cases)]
                for index in range(requested)
            )

            request = self._prompt_builder.corner_cover(
                context,
                selected,
                distribution=distribution,
                accepted_examples=accepted_examples,
                reference_examples=reference_examples,
                valid_outputs_per_example=valid_outputs_per_example,
            )

            raw = self._call_model(
                request,
                context.spec.task,
                with_axis_tags=True,
                valid_outputs_per_example=valid_outputs_per_example,
            )

            self._cache_axis_tags(tag_cache, raw)
            return raw

        result = pipeline.run(
            producer=producer,
            context=context,
            target_n=target_n,
            judge=apply_judge,
            is_corner=True,
            reset_deduplicator=reset_deduplicator,
        )

        return result, tag_cache

    @staticmethod
    def _cache_axis_tags(
            cache: dict[tuple[str, str], dict[str, str]],
            raw_examples: Sequence[Any],
    ) -> None:
        for raw in raw_examples:
            if isinstance(raw, BaseModel):
                payload = raw.model_dump()
            elif isinstance(raw, dict):
                payload = raw
            else:
                payload = {
                    "input": getattr(raw, "input", ""),
                    "output": getattr(raw, "output", ""),
                    "axis_tags": getattr(raw, "axis_tags", {}),
                }

            input_key = str(payload.get("input", "")).strip().casefold()
            output_key = str(payload.get("output", "")).strip().casefold()
            raw_tags = payload.get("axis_tags") or {}
            if input_key and isinstance(raw_tags, dict):
                cache[(input_key, output_key)] = {
                    str(axis): str(value)
                    for axis, value in raw_tags.items()
                    if isinstance(value, str)
                }

    @staticmethod
    def _record_feedback_batch(
            state: GenerationState,
            distribution: TaskDistribution,
            context: GenerationContext,
            examples: Sequence[Example],
            tag_cache: dict[tuple[str, str], dict[str, str]],
    ) -> None:
        for example in examples:
            key = (example.input.strip().casefold(), example.output.strip().casefold())
            tags = validate_axis_tags(
                distribution,
                tag_cache.get(key),
                input=example.input,
                output=example.output,
                spec=context.spec,
            )
            state.record(tags)

    @property
    def last_distribution(self) -> TaskDistribution | None:
        """TaskDistribution from the most recent generate() call, for diagnostics."""

        return self._last_distribution

    @property
    def last_generation_state(self) -> GenerationState | None:
        """Final feedback coverage state from the most recent generate() call."""

        return self._last_generation_state

    def _build_pipeline(
            self,
            *,
            novelty: bool,
            min_references: int = 0,
    ) -> ValidationPipeline:
        return ValidationPipeline(
            validator=ExampleValidator(min_references=min_references),
            deduplicator=Deduplicator(
                enable_semantic_novelty=novelty,
                enable_structural_novelty=novelty,
            ),
            judge=LLMJudge(
                self._judge_model,
                quality_threshold=self._judge_quality_threshold,
                batch_size=self._judge_batch_size,
                retry_config=self._retry_config,
            ),
            max_topup_attempts=self._max_topup_attempts,
        )
