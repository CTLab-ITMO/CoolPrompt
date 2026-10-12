"""High-level orchestration for synthetic-data generation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Sequence
from html import unescape
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel

from coolprompt.spec_generator.distribution import (
    GenerationState,
    TaggedGenerationBatch,
    TaskDistribution,
    _TaskDistributionBuilder,
    axis_quotas,
    build_generation_targets,
    coverage_gaps,
    trim_indices,
    validate_axis_tags,
)
from coolprompt.spec_generator.schemas import TaskExamples
from coolprompt.spec_generator.models import (
    Example,
    GenerationContext,
    GenerationResult,
    TaskSpecDraft,
)
from coolprompt.spec_generator.prompt_builder import GenerationPromptBuilder
from coolprompt.spec_generator.spec_builder import SpecBuilder
from coolprompt.spec_generator.utils.model_utils import invoke_structured
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.spec_generator.validation.format import Deduplicator, ExampleValidator
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger

_OUTPUT_SCHEMAS: dict[Task, type[BaseModel]] = {
    Task.CLASSIFICATION: TaskExamples,
    Task.GENERATION: TaskExamples,
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

    if type(num_samples) is not int or not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 100")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be at least 1")


def _extract_examples(payload: Any) -> list[Any]:
    """Extract a non-empty examples list from a model response."""

    examples = (
        getattr(payload, "examples", None)
        if isinstance(payload, BaseModel)
        else payload.get("examples") if isinstance(payload, dict) else None
    )

    if not isinstance(examples, list):
        raise GenerationResponseError(
            "Generation response does not contain an examples list."
        )

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
        distribution_model: BaseLanguageModel | None = None,
    ) -> None:
        """Initialize generation, specification, and distribution components."""

        if type(max_topup_attempts) is not int or max_topup_attempts < 1:
            raise ValueError("max_topup_attempts must be a positive integer")
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
            model=model if distribution_model is None else distribution_model,
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
        task_distribution: TaskDistribution | None = None,
        detect_dataset: bool = False,
        num_samples: int = 40,
        batch_size: int = 15,
        use_task_distribution: bool = True,
        feedback_controlled: bool = True,
        strict_coverage: bool = False,
    ) -> GenerationResult:
        """Generate exactly ``num_samples`` synthetic examples."""

        _validate_generation_args(num_samples, batch_size)

        if feedback_controlled and not use_task_distribution:
            raise ValueError("feedback_controlled requires use_task_distribution=True")
        if strict_coverage and not feedback_controlled:
            raise ValueError("strict_coverage requires feedback_controlled=True")

        self._last_distribution = None
        self._last_generation_state = None

        context = self.build_context(
            prompt,
            dataset_name,
            draft=draft,
            examples=examples,
            detect_dataset=detect_dataset,
        )
        self._validate_context(context)

        distribution = self._resolve_distribution(
            prompt=prompt,
            context=context,
            distribution=task_distribution,
            enabled=use_task_distribution,
        )

        logger.info(
            "TaskDistribution:\n%s",
            (
                distribution.model_dump_json(indent=2)
                if distribution is not None
                else "None"
            ),
        )

        self._last_distribution = distribution

        if feedback_controlled:
            assert distribution is not None

            generated = self._generate_feedback_controlled(
                context=context,
                distribution=distribution,
                num_samples=num_samples,
                batch_size=batch_size,
                strict_coverage=strict_coverage,
            )
        else:
            generated = self._generate_validated(
                context,
                num_samples,
                batch_size,
                distribution,
            )

        if len(generated) != num_samples:
            raise RuntimeError(
                f"Expected {num_samples} examples, received {len(generated)}"
            )

        return GenerationResult(
            examples=tuple(generated),
            context=context,
        )

    def _resolve_distribution(
        self,
        *,
        prompt: str,
        context: GenerationContext,
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
        )

    @staticmethod
    def _validate_context(context: GenerationContext) -> None:
        """Reject task types unsupported by the generation schemas."""

        if context.spec.task not in _OUTPUT_SCHEMAS:
            supported = ", ".join(task.value for task in _OUTPUT_SCHEMAS)

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
        """Generate and validate exactly the requested examples."""

        if target <= 0:
            return []

        result = self._build_pipeline().run(
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
            raise RuntimeError(
                f"Could not generate enough examples: {len(result)}/{target}"
            )

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

        schema = TaggedGenerationBatch if with_axis_tags else _OUTPUT_SCHEMAS[task]

        def invoke() -> list[Any]:
            """Perform one retryable model invocation and extract its examples."""

            output = invoke_structured(
                self._model,
                request,
                schema.model_json_schema(),
                method="function_calling" if with_axis_tags else "json_schema",
                error_cls=GenerationResponseError,
                label="Generation",
            )

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
        strict_coverage: bool,
    ) -> list[Example]:
        """Generate examples while targeting coverage in the final dataset."""

        if num_samples <= 20:
            oversample_factor = 1.5
        elif num_samples <= 50:
            oversample_factor = 1.3
        else:
            oversample_factor = 1.2

        limit = math.ceil(num_samples * oversample_factor)

        pipeline = self._build_pipeline()
        quotas = axis_quotas(distribution, num_samples)

        state = GenerationState()
        accepted: list[Example] = []
        tags: list[dict[str, str]] = []

        def gaps_of(generation_state: GenerationState) -> list[dict]:
            return coverage_gaps(distribution, generation_state, num_samples)[0]

        def gap_total(gap_items: Sequence[dict]) -> int:
            return sum(max(0, int(gap.get("gap", 0))) for gap in gap_items)

        def projected() -> tuple[set[int], GenerationState]:
            """Best final-size subset (indices to drop) and its coverage state."""

            excess = len(accepted) - num_samples
            if excess <= 0:
                return set(), state

            drop_indices = trim_indices(
                tags, excess, quotas, state.axis_counts, num_samples
            )

            kept = GenerationState()
            for index, item in enumerate(tags):
                if index not in drop_indices:
                    kept.record(item)

            return drop_indices, kept

        def step(
            target_n: int,
            target_state: GenerationState,
            *,
            first: bool,
            is_post_target: bool,
        ) -> bool:
            targets, avoid = (
                (None, ())
                if first
                else build_generation_targets(
                    distribution,
                    target_state,
                    batch_size=target_n,
                    remaining_budget=target_n,
                    total_target=num_samples,
                    post_target=is_post_target,
                )
            )

            if is_post_target and not targets:
                return False

            batch, cache = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=target_n,
                batch_size=batch_size,
                reset_deduplicator=first,
                targets=targets,
                avoid=avoid,
                accepted_examples=accepted,
            )

            for example in batch:
                example_tags = validate_axis_tags(
                    distribution,
                    cache.get(self._example_key(example.input, example.output)),
                    output=example.output,
                    spec=context.spec,
                )
                state.record(example_tags)
                tags.append(example_tags)

            accepted.extend(batch)

            logger.info("Progress: accepted=%d/%d", len(accepted), num_samples)

            return bool(batch)

        stalls = 0

        while len(accepted) < limit:
            _, current_state = projected()
            gaps = gaps_of(current_state)
            before = gap_total(gaps)
            post_target = len(accepted) >= num_samples

            if post_target and not gaps:
                break

            remaining = limit - len(accepted)

            if post_target:
                n = max(1, min(batch_size, remaining, math.ceil(before * 2.0)))
            else:
                n = min(batch_size, num_samples - len(accepted))

            produced = step(
                n, current_state, first=not accepted, is_post_target=post_target
            )

            if post_target:
                after = gap_total(gaps_of(projected()[1]))
                stalls = 0 if after < before else stalls + 1

                logger.info("Gap batch: n=%d gap %d -> %d", n, before, after)

                if stalls >= 3:
                    break

                continue

            if not produced:
                break

        if len(accepted) < num_samples:
            self._last_generation_state = state
            raise RuntimeError(
                f"Not enough examples generated: {len(accepted)}/{num_samples}"
            )

        drop, state = projected()

        accepted = [
            example for index, example in enumerate(accepted) if index not in drop
        ]

        self._last_generation_state = state

        if gaps := gaps_of(state):
            message = f"Coverage incomplete after {len(accepted)} examples: {gaps}"

            if strict_coverage:
                raise RuntimeError(message)

            logger.warning(message)

        return accepted

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
    ) -> tuple[list[Example], dict[tuple[str, str], dict[str, str]]]:
        """Generate, validate, and retain axis tags for one feedback batch."""

        tag_cache: dict[tuple[str, str], dict[str, str]] = {}
        tag_queue: dict[tuple[str, str], deque[dict[str, str]]] = {}
        local_accepted: list[Example] = []
        pending = [dict(target) for target in (targets or [])]
        required_axes = {axis.name for axis in distribution.axes}
        candidate_tags: dict[str, str] = {}

        def matching_target(tags: dict[str, str]) -> dict[str, Any] | None:
            for candidate_target in sorted(
                pending,
                key=lambda candidate: len(candidate["constraints"]),
                reverse=True,
            ):
                if candidate_target["count"] <= 0:
                    continue

                if all(
                    tags.get(item["axis"]) == item["value_id"]
                    for item in candidate_target["constraints"]
                ):
                    return candidate_target

            return None

        def accept_candidate(example: Example) -> bool:
            nonlocal candidate_tags
            key = self._example_key(example.input, example.output)
            queued = tag_queue.get(key)
            candidate_tags = validate_axis_tags(
                distribution,
                queued.popleft() if queued else None,
                output=example.output,
                spec=context.spec,
            )
            tags = candidate_tags
            valid = required_axes == set(tags) and (
                targets is None or matching_target(tags) is not None
            )

            if not valid:
                logger.info(
                    "Rejected incomplete tags or unmet coverage target: %s",
                    example.input,
                )

            return valid

        def on_accept(example: Example) -> None:
            if targets is not None:
                target = matching_target(candidate_tags)

                if target is None:
                    raise RuntimeError(
                        "Accepted example does not match an open generation target."
                    )
                target["count"] -= 1

            tag_cache[self._example_key(example.input, example.output)] = candidate_tags
            local_accepted.append(example)

        common = {
            "accepted_examples": accepted_examples,
        }

        def proposal_targets(proposal_n: int) -> list[dict[str, Any]]:
            """Spread the extra quota across open targets to cover an oversampled batch."""

            active_targets = [
                {**target, "constraints": [*target["constraints"]]}
                for target in pending
                if target["count"] > 0
            ]

            if not active_targets:
                return []

            current_total = sum(target["count"] for target in active_targets)
            extra = max(0, proposal_n - current_total)

            share, rest = divmod(extra, len(active_targets))

            for index, target in enumerate(active_targets):
                target["count"] += share + int(index < rest)

            return active_targets

        def producer(remaining: int) -> list[Any]:
            """Generate a proposal pool, oversampling small top-up requests."""
            common["accepted_examples"] = [*accepted_examples, *local_accepted]

            proposal_n = min(batch_size, max(remaining, 5))
            args = (context, proposal_n, distribution)

            request = (
                self._prompt_builder.distribution_aware(*args, **common)
                if targets is None
                else self._prompt_builder.targeted(
                    *args,
                    targets=proposal_targets(proposal_n),
                    avoid=avoid,
                    **common,
                )
            )

            try:
                raw = self._call_model(
                    request,
                    context.spec.task,
                    with_axis_tags=True,
                )
            except GenerationResponseError as exc:
                logger.warning(
                    "Generation response was invalid; treating batch as empty: %s", exc
                )
                return []

            self._queue_axis_tags(tag_queue, raw)
            return raw

        return (
            pipeline.run(
                producer=producer,
                context=context,
                target_n=target_n,
                reset_deduplicator=reset_deduplicator,
                accept_candidate=accept_candidate,
                on_accept=on_accept,
            ),
            tag_cache,
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

    @staticmethod
    def _example_key(input_: str, output: str) -> tuple[str, str]:
        """Use the same input decoding before and after validation."""
        return unescape(input_).strip().casefold(), output.strip().casefold()

    @classmethod
    def _queue_axis_tags(
        cls,
        queue: dict[tuple[str, str], deque[dict[str, str]]],
        raw_examples: Sequence[Any],
    ) -> None:
        """Preserve each candidate's own tags, including duplicate text pairs."""

        for raw in raw_examples:
            payload = cls._payload(raw)
            input_, output, tags = (
                payload.get("input"),
                payload.get("output"),
                payload.get("axis_tags"),
            )

            if not (isinstance(input_, str) and input_.strip()):
                continue
            if not isinstance(output, str):
                continue

            queue.setdefault(cls._example_key(input_, output), deque()).append(
                {
                    str(axis): value
                    for axis, value in (tags.items() if isinstance(tags, dict) else ())
                    if isinstance(value, str)
                }
            )

    @property
    def last_distribution(self) -> TaskDistribution | None:
        """TaskDistribution from the most recent generate() call."""
        return self._last_distribution

    @property
    def last_generation_state(self) -> GenerationState | None:
        """Final feedback coverage state from the most recent generate() call."""
        return self._last_generation_state

    def _build_pipeline(self) -> ValidationPipeline:
        """Create a fresh validation pipeline for one generation phase."""

        return ValidationPipeline(
            validator=ExampleValidator(),
            deduplicator=Deduplicator(),
            max_topup_attempts=self._max_topup_attempts,
        )
