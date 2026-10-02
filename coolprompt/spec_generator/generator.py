"""High-level orchestration for synthetic-data generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from html import unescape
from typing import Any, NoReturn

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.spec_generator.distribution import (
    GenerationState,
    TaggedGenerationBatch,
    TaskDistribution,
    _TaskDistributionBuilder,
    build_generation_targets,
    coverage_gaps,
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
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.spec_generator.validation.format import Deduplicator, ExampleValidator
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_json
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

    if isinstance(payload, AIMessage):
        payload = payload.content
    if isinstance(payload, str):
        payload = extract_json(payload)

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
    ) -> GenerationResult:
        """Generate exactly ``num_samples`` synthetic examples."""

        _validate_generation_args(num_samples, batch_size)

        if feedback_controlled and not use_task_distribution:
            raise ValueError("feedback_controlled requires use_task_distribution=True")

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
            examples=tuple(map(self._coerce_example, generated)),
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
        chat_model = resolve_chat_model(self._model)

        def invoke() -> list[Any]:
            """Perform one retryable model invocation and extract its examples."""

            if chat_model is None:
                output = self._model.invoke(request)
            else:
                method = "function_calling" if with_axis_tags else "json_schema"
                output = chat_model.with_structured_output(
                    schema=schema.model_json_schema(), method=method
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
    ) -> list[Example]:
        """Generate examples while preserving requested distribution coverage."""
        max_extra = 10
        limit = num_samples + max_extra
        pipeline = self._build_pipeline()

        state = GenerationState()
        accepted: list[Example] = []
        accepted_tags: list[dict[str, str]] = []  # strictly 1:1 with `accepted`

        def tags_for(
            example: Example,
            tag_cache: dict[tuple[str, str], dict[str, str]],
        ) -> dict[str, str]:
            key = self._example_key(example.input, example.output)
            return validate_axis_tags(
                distribution,
                tag_cache.get(key),
                output=example.output,
                spec=context.spec,
            )

        def record(
            st: GenerationState,
            examples: Sequence[Example],
            tag_cache: dict[tuple[str, str], dict[str, str]],
        ) -> list[dict[str, str]]:
            aligned = [tags_for(example, tag_cache) for example in examples]
            for tags in aligned:
                st.record(tags)
            return aligned

        def rebuild_state(tags: Sequence[dict[str, str]]) -> GenerationState:
            rebuilt = GenerationState()
            for item in tags:
                rebuilt.record(item)
            return rebuilt

        def gaps_of(st: GenerationState) -> list[dict]:
            return coverage_gaps(distribution, st, num_samples)[0]

        def fail(message: str) -> NoReturn:
            self._last_generation_state = state
            raise RuntimeError(message)

        def step(n: int, budget: int | None = None) -> bool:
            """Run one batch. budget=None means the initial unconstrained batch."""
            first = budget is None
            if first:
                targets, avoid = None, ()
            else:
                targets, avoid = build_generation_targets(
                    distribution,
                    state,
                    batch_size=n,
                    remaining_budget=budget,
                    total_target=num_samples,
                )
                logger.info("Generation targets: %s", targets)

            batch, tag_cache = self._run_feedback_batch(
                pipeline=pipeline,
                context=context,
                distribution=distribution,
                target_n=n,
                batch_size=batch_size,
                reset_deduplicator=first,
                targets=targets,
                avoid=avoid,
                accepted_examples=accepted,
            )
            if not batch:
                return False

            batch_tags = record(state, batch, tag_cache)
            accepted.extend(batch)
            accepted_tags.extend(batch_tags)
            logger.info("Progress: accepted=%d/%d", len(accepted), num_samples)
            return True

        if not step(min(batch_size, num_samples)):
            fail("Initial generation batch returned no examples.")

        while len(accepted) < num_samples:
            remaining = num_samples - len(accepted)
            if not step(min(batch_size, remaining), remaining):
                break

        if len(accepted) < num_samples:
            fail(f"Not enough examples generated: {len(accepted)}/{num_samples}")

        gaps = gaps_of(state)
        while gaps and len(accepted) < limit:
            logger.info("Coverage gaps remain: %s", gaps)
            gap_size = sum(max(0, int(g.get("gap", 0))) for g in gaps)
            n = min(batch_size, limit - len(accepted), max(1, gap_size))
            if not step(n, n):
                break
            gaps = gaps_of(state)

        if gaps:
            fail(f"Coverage incomplete after {len(accepted)} examples: {gaps}")

        while len(accepted) > num_samples:
            for i in range(len(accepted)):
                trial = rebuild_state(accepted_tags[:i] + accepted_tags[i + 1 :])
                if gaps_of(trial):
                    continue
                del accepted[i], accepted_tags[i]
                state = trial
                logger.info("Retained %d/%d examples", len(accepted), num_samples)
                break
            else:
                fail(f"Could not trim to {num_samples} without coverage gaps.")

        final_gaps = gaps_of(state)
        if len(accepted) != num_samples or final_gaps:
            fail(
                f"Generation incomplete: {len(accepted)}/{num_samples}, gaps={final_gaps}"
            )

        self._last_generation_state = state
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
        local_accepted: list[Example] = []
        pending = [dict(target) for target in (targets or [])]
        required_axes = {axis.name for axis in distribution.axes}

        def example_tags(example: Example) -> dict[str, str]:
            key = self._example_key(example.input, example.output)
            return validate_axis_tags(
                distribution,
                tag_cache.get(key),
                output=example.output,
                spec=context.spec,
            )

        def matching_target(tags: dict[str, str]) -> dict[str, Any] | None:
            for target in sorted(
                pending,
                key=lambda target: len(target["constraints"]),
                reverse=True,
            ):
                if target["count"] <= 0:
                    continue

                if all(
                    tags.get(item["axis"]) == item["value_id"]
                    for item in target["constraints"]
                ):
                    return target

            return None

        def accept_candidate(example: Example) -> bool:
            tags = example_tags(example)
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
                target = matching_target(example_tags(example))

                if target is None:
                    raise RuntimeError(
                        "Accepted example does not match an open generation target."
                    )
                target["count"] -= 1

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

            raw = self._call_model(request, context.spec.task, with_axis_tags=True)
            self._cache_axis_tags(tag_cache, raw)
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
    def _cache_axis_tags(
        cls,
        cache: dict[tuple[str, str], dict[str, str]],
        raw_examples: Sequence[Any],
    ) -> None:
        """Cache valid axis tags by normalized input-output pair."""

        for raw in raw_examples:
            payload = cls._payload(raw)
            input_, output, tags = (
                payload.get("input"),
                payload.get("output"),
                payload.get("axis_tags"),
            )

            if not (isinstance(input_, str) and input_.strip()):
                continue
            if not isinstance(output, str) or not isinstance(tags, dict):
                continue

            cache[cls._example_key(input_, output)] = {
                str(axis): value
                for axis, value in tags.items()
                if isinstance(value, str)
            }

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
