"""High-level orchestration for synthetic-data generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

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
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_json

_OUTPUT_SCHEMAS: dict[Task, type[BaseModel]] = {
    Task.CLASSIFICATION: ClassificationTaskStructuredOutputSchema,
    Task.GENERATION: GenerationTaskStructuredOutputSchema,
}


class GenerationResponseError(ValueError):
    """Raised when a generation response cannot be used safely."""


def _batch_sizes(total: int, batch_size: int) -> Iterator[int]:
    """Yield batch sizes that sum to the requested total."""

    while total > 0:
        yield min(total, batch_size)
        total -= batch_size


def _validate_generation_args(num_samples: int, batch_size: int) -> None:
    """Validate public generation arguments."""

    if not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 100")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")


def _extract_examples(payload: Any) -> list[Any]:
    """Extract a non-empty examples list from a model response."""

    if isinstance(payload, AIMessage):
        payload = payload.content
    if isinstance(payload, str):
        payload = extract_json(payload)

    examples = (
        getattr(payload, "examples", None)
        if isinstance(payload, BaseModel)
        else payload.get("examples")
        if isinstance(payload, dict)
        else None
    )

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
            *,
            task_spec_model: BaseLanguageModel | None = None,
    ) -> None:
        """Initialize generation, specification, and distribution components."""

        self._model = model
        self._retry_config = retry_config or RetryConfig()
        self._max_topup_attempts = max_topup_attempts

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
            dataset_name: str | None = None,
            *,
            draft: TaskSpecDraft | None = None,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            detect_dataset: bool = False,
    ) -> GenerationContext:
        """Build the validated context used by subsequent generation stages."""

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
            dataset_name: str | None = None,
            *,
            draft: TaskSpecDraft | None = None,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            distribution_examples: Sequence[tuple[str, str] | Example] | None = None,
            task_distribution: TaskDistribution | None = None,
            detect_dataset: bool = True,
            num_samples: int = 40,
            batch_size: int = 15,
            structural_validation: bool = False,
            use_task_distribution: bool = True,
            feedback_controlled: bool = True
    ) -> GenerationResult:
        """Generate exactly ``num_samples`` synthetic examples."""

        _validate_generation_args(num_samples, batch_size)

        if feedback_controlled and not use_task_distribution:
            raise ValueError("feedback_controlled requires use_task_distribution=True")

        context = self.build_context(
            prompt,
            dataset_name,
            draft=draft,
            examples=examples,
            detect_dataset=detect_dataset,
        )

        self._validate_context(context)

        reference_examples = self._reference_examples(distribution_examples, fallback=context.seed_examples)

        distribution = self._resolve_distribution(
            prompt=prompt,
            context=context,
            reference_examples=reference_examples,
            distribution=task_distribution,
            enabled=use_task_distribution,
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
                reference_examples=reference_examples,
                structural_validation=structural_validation,
            )
        elif structural_validation:
            generated = self._generate_validated(
                context,
                num_samples,
                batch_size,
                distribution,
            )
        else:
            generated = self._generate_group(
                context,
                num_samples,
                batch_size,
                distribution=distribution,
            )

        if len(generated) != num_samples:
            raise RuntimeError(f"Expected {num_samples} examples, received {len(generated)}")

        return GenerationResult(
            examples=tuple(map(self._coerce_example, generated)),
            context=context
        )

    @staticmethod
    def _reference_examples(
            examples: Sequence[tuple[str, str] | Example] | None,
            *,
            fallback: Sequence[Example],
    ) -> tuple[Example, ...]:
        """Normalize explicit distribution references or use seed examples."""

        if not examples:
            return tuple(fallback)

        return tuple(
            item if isinstance(item, Example) else Example(input=item[0], output=item[1])
            for item in examples
        )

    def _resolve_distribution(
            self,
            *,
            prompt: str,
            context: GenerationContext,
            reference_examples: Sequence[Example],
            distribution: TaskDistribution | None,
            enabled: bool,
    ) -> TaskDistribution | None:
        """Return a supplied or inferred distribution when the feature is enabled."""

        if not enabled:
            return None

        if distribution is not None:
            return distribution

        return self._distribution_builder.build(
            prompt=prompt,
            spec=context.spec,
            examples=context.seed_examples,
            reference_examples=reference_examples,
        )

    @classmethod
    def _coerce_example(cls, item: Any) -> Example:
        """Convert a generated payload into the public Example model."""

        if isinstance(item, Example):
            return item

        payload = cls._payload(item)
        return Example(
            input=payload["input"],
            output=payload["output"],
        )

    @staticmethod
    def _validate_context(context: GenerationContext) -> None:
        """Reject task types unsupported by the generation schemas."""

        if context.spec.task not in _OUTPUT_SCHEMAS:
            supported = ", ".join(
                task.value
                for task in _OUTPUT_SCHEMAS
            )

            raise ValueError(
                f"Unsupported task {context.spec.task!r}; "
                f"supported tasks: {supported}"
            )

    def _generate_validated(
            self,
            context: GenerationContext,
            target: int,
            batch_size: int,
            distribution: TaskDistribution | None = None,
    ) -> list[Example]:
        """Generate and structurally validate exactly the requested examples."""

        if target <= 0:
            return []

        result = self._build_pipeline(novelty=True).run(
            producer=lambda remaining: self._generate_group(
                context,
                remaining,
                batch_size,
                distribution=distribution,
            ),
            context=context,
            target_n=target,
            reset_deduplicator=True,
        )

        if len(result) < target:
            raise RuntimeError(f"Could not generate enough examples: {len(result)}/{target}")

        return result

    def _generate_group(
            self,
            context: GenerationContext,
            total: int,
            batch_size: int,
            *,
            distribution: TaskDistribution | None = None,
    ) -> list[Any]:
        """Generate examples in bounded batches with optional distribution guidance."""

        generated: list[Any] = []

        for size in _batch_sizes(total, batch_size):
            request = (
                self._prompt_builder.regular(context, size)
                if distribution is None
                else self._prompt_builder.distribution_aware(
                    context,
                    size,
                    distribution,
                )
            )
            generated.extend(
                self._call_model(
                    request,
                    context.spec.task,
                    with_axis_tags=distribution is not None,
                )
            )

        return generated

    def _call_model(
            self,
            request: str,
            task: Task,
            *,
            with_axis_tags: bool = False,
    ) -> list[Any]:
        """Invoke the model with the appropriate structured-output schema."""

        schema = (
            TaggedGenerationBatch
            if with_axis_tags
            else _OUTPUT_SCHEMAS[task]
        )
        chat_model = resolve_chat_model(self._model)

        def invoke() -> list[Any]:
            """Perform one retryable model invocation and extract its examples."""

            if chat_model is None:
                output = self._model.invoke(request)
            else:
                method = (
                    "function_calling"
                    if with_axis_tags
                    else "json_schema"
                )
                output = chat_model.with_structured_output(schema=schema, method=method).invoke(request)

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
            reference_examples: Sequence[Example],
            structural_validation: bool,
    ) -> list[Example]:
        """Generate, observe coverage, then target the next batch."""

        pipeline = self._build_pipeline(novelty=structural_validation)
        state = GenerationState()
        accepted: list[Example] = []

        first_n = min(batch_size, num_samples)

        batch, tags = self._run_feedback_batch(
            pipeline=pipeline,
            context=context,
            distribution=distribution,
            target_n=first_n,
            batch_size=batch_size,
            reset_deduplicator=True,
            targets=None,
            avoid=(),
            accepted_examples=accepted,
            reference_examples=reference_examples,
        )
        accepted.extend(batch)
        self._record_feedback_batch(
            state,
            distribution,
            context,
            batch,
            tags,
        )

        while len(accepted) < num_samples:
            remaining = num_samples - len(accepted)
            current_n = min(batch_size, remaining)

            targets, avoid = build_generation_targets(
                distribution,
                state,
                batch_size=current_n,
                remaining_budget=remaining,
                total_target=num_samples,
            )

            batch, tags = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=current_n,
                batch_size=batch_size,
                reset_deduplicator=False,
                targets=targets,
                avoid=avoid,
                accepted_examples=accepted,
                reference_examples=reference_examples,
            )

            if not batch:
                break

            accepted.extend(batch)
            self._record_feedback_batch(
                state,
                distribution,
                context,
                batch,
                tags,
            )

        if len(accepted) < num_samples:
            raise RuntimeError(
                "Could not generate enough feedback-controlled examples: "
                f"{len(accepted)}/{num_samples}"
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
            reset_deduplicator: bool,
            targets: Sequence[dict[str, Any]] | None,
            avoid: Sequence[dict[str, Any]],
            accepted_examples: Sequence[Example],
            reference_examples: Sequence[Example],
    ) -> tuple[list[Example], dict[tuple[str, str], dict[str, str]]]:
        """Generate, validate, and retain axis tags for one feedback batch."""

        tag_cache: dict[tuple[str, str], dict[str, str]] = {}
        common = {
            "accepted_examples": accepted_examples,
            "reference_examples": reference_examples,
        }

        def producer(remaining: int) -> list[Any]:
            """Generate the next batch, targeting coverage gaps when available."""
            args = context, remaining, distribution

            if targets is None:
                request = self._prompt_builder.distribution_aware(*args, **common)
            else:
                request = self._prompt_builder.targeted(
                    *args,
                    targets=targets,
                    avoid=avoid,
                    **common,
                )

            raw = self._call_model(request, context.spec.task, with_axis_tags=True)
            self._cache_axis_tags(tag_cache, raw)
            return raw

        return (
            pipeline.run(
                producer=producer,
                context=context,
                target_n=target_n,
                reset_deduplicator=reset_deduplicator,
            ),
            tag_cache
        )

    @staticmethod
    def _payload(raw: Any) -> dict[str, Any]:
        """Convert an arbitrary generated item into a dictionary payload."""

        if isinstance(raw, BaseModel):
            return raw.model_dump()

        if isinstance(raw, dict):
            return raw

        return {
            "input": getattr(raw, "input", ""),
            "output": getattr(raw, "output", ""),
            "axis_tags": getattr(raw, "axis_tags", {}),
        }

    @classmethod
    def _cache_axis_tags(
            cls,
            cache: dict[tuple[str, str], dict[str, str]],
            raw_examples: Sequence[Any],
    ) -> None:
        """Index valid model-provided axis tags by normalized input-output pair."""

        for raw in raw_examples:
            payload = cls._payload(raw)

            input_ = str(payload.get("input", "")).strip().casefold()
            output_ = str(payload.get("output", "")).strip().casefold()
            tags = payload.get("axis_tags")

            if not input_ or not isinstance(tags, dict):
                continue

            cache[input_, output_] = {
                str(axis): value
                for axis, value in tags.items()
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
        """Validate batch tags and record their observed coverage counts."""

        for example in examples:
            key = (
                example.input.strip().casefold(),
                example.output.strip().casefold(),
            )
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
        """TaskDistribution from the most recent generate() call."""
        return self._last_distribution

    @property
    def last_generation_state(self) -> GenerationState | None:
        """Final feedback coverage state from the most recent generate() call."""
        return self._last_generation_state

    def _build_pipeline(self, *, novelty: bool) -> ValidationPipeline:
        """Create a fresh validation pipeline for one generation phase."""

        return ValidationPipeline(
            validator=ExampleValidator(),
            deduplicator=Deduplicator(
                enable_semantic_novelty=novelty,
                enable_structural_novelty=novelty,
            ),
            max_topup_attempts=self._max_topup_attempts,
        )
