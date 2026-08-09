"""High-level orchestration for synthetic-data generation."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.data_generator.pydantic_formatters import (
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskStructuredOutputSchema,
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


def _split_count(total: int, corner_ratio: float) -> tuple[int, int]:
    """Split the total into regular and corner-case counts."""

    corner = int(total * corner_ratio)
    return total - corner, corner


def _batch_sizes(total: int, batch_size: int) -> Iterator[int]:
    """Yield batch sizes until the requested total is reached."""

    remaining = total
    while remaining > 0:
        current = min(remaining, batch_size)
        yield current
        remaining -= current


def _validate_generation_args(
        num_samples: int,
        batch_size: int,
        corner_ratio: float,
) -> None:
    """Validate synthetic-generation arguments."""

    if not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 100")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= corner_ratio <= 1.0:
        raise ValueError("corner_ratio must be between 0.0 and 1.0")


def _extract_examples(payload: Any) -> list[Any]:
    """Extract examples from a model response."""

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
            max_topup_attempts: int = 3,
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

    def build_context(
            self,
            prompt: str,
            *,
            draft: TaskSpecDraft | None = None,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            detect_dataset: bool = False,
            dataset_name: str | None = None,
    ) -> GenerationContext:
        """Build a TaskSpec without generating examples."""

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
            detect_dataset: bool = False,
            num_samples: int = 8,
            batch_size: int = 15,
            corner_ratio: float = 0.4,
            structural_validation: bool = True,
            judge_regular: bool = False,
            judge_corner_cases: bool = False,
    ) -> GenerationResult:
        """Generate exactly ``num_samples`` synthetic examples."""

        _validate_generation_args(
            num_samples,
            batch_size,
            corner_ratio,
        )

        context = self.build_context(
            prompt,
            draft=draft,
            examples=examples,
            detect_dataset=detect_dataset,
        )

        self._validate_context(context)

        regular_count, corner_count = _split_count(
            num_samples,
            corner_ratio,
        )

        if (
                not structural_validation
                and (judge_regular or judge_corner_cases)
        ):
            raise ValueError("LLM judging requires structural_validation=True")

        if structural_validation:
            generated = self._generate_validated(
                context=context,
                regular_count=regular_count,
                corner_count=corner_count,
                batch_size=batch_size,
                judge_regular=judge_regular,
                judge_corner_cases=judge_corner_cases,
            )
        else:
            generated = self._generate_unvalidated(
                context=context,
                regular_count=regular_count,
                corner_count=corner_count,
                batch_size=batch_size,
            )

        if len(generated) != num_samples:
            raise RuntimeError(
                f"Expected {num_samples} examples, "
                f"received {len(generated)}"
            )

        return GenerationResult(
            examples=tuple(
                item
                if isinstance(item, Example)
                else Example.model_validate(item)
                for item in generated
            ),
            context=context,
        )

    @staticmethod
    def _validate_context(context: GenerationContext) -> None:
        """Validate that the context uses a supported task type."""

        if context.spec.task not in _OUTPUT_SCHEMAS:
            supported = ", ".join(task.value for task in _OUTPUT_SCHEMAS)
            raise ValueError(f"Unsupported task {context.spec.task!r}; supported tasks: {supported}")

    @staticmethod
    def _with_examples(
            context: GenerationContext,
            examples: Sequence[tuple[str, str] | Example],
    ) -> GenerationContext:
        """Return a context containing the provided seed examples."""

        seed_examples = tuple(
            item
            if isinstance(item, Example)
            else Example(input=item[0], output=item[1])
            for item in examples
        )
        payload = context.model_dump()
        payload["seed_examples"] = seed_examples
        return GenerationContext.model_validate(payload)

    def _generate_validated(
            self,
            context: GenerationContext,
            regular_count: int,
            corner_count: int,
            batch_size: int,
            judge_regular: bool,
            judge_corner_cases: bool,
    ) -> list[Example]:
        """Generate and validate regular and corner-case examples."""

        pipeline = self._build_pipeline()

        regular = self._generate_validated_group(
            pipeline,
            context,
            regular_count,
            batch_size,
            is_corner=False,
            apply_judge=judge_regular,
            reset_deduplicator=True,
        )
        corner = self._generate_validated_group(
            pipeline,
            context,
            corner_count,
            batch_size,
            is_corner=True,
            apply_judge=judge_corner_cases,
            reset_deduplicator=not regular,
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
    ) -> list[Example]:
        """Generate one validated example group."""

        if target <= 0:
            return []

        result = pipeline.run(
            producer=lambda remaining: self._generate_group(
                context,
                remaining,
                batch_size,
                is_corner=is_corner,
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
    ) -> list[Any]:
        """Generate examples without structural validation."""

        regular = self._generate_group(
            context,
            regular_count,
            batch_size,
            is_corner=False,
        )
        corner = self._generate_group(
            context,
            corner_count,
            batch_size,
            is_corner=True,
        )
        return regular + corner

    def _generate_group(
            self,
            context: GenerationContext,
            total: int,
            batch_size: int,
            *,
            is_corner: bool,
    ) -> list[Any]:
        """Generate the requested number of regular or corner-case examples in batches."""

        generated: list[Any] = []

        for size in _batch_sizes(total, batch_size):
            request = self._build_request(context, size, is_corner=is_corner)
            generated.extend(self._call_model(request, context.spec.task))

        return generated

    def _build_request(
            self,
            context: GenerationContext,
            n: int,
            *,
            is_corner: bool,
    ) -> str:
        """Build a prompt for generating regular or corner-case examples."""

        if not is_corner:
            return self._prompt_builder.regular(context, n)

        cases = context.spec.corner_cases
        selected = random.sample(cases, min(len(cases), n))
        return self._prompt_builder.corner(
            context,
            n,
            corner_cases=selected,
        )

    def _call_model(self, request: str, task: Task) -> list[Any]:
        """Invoke the model with the task-specific schema and return generated examples."""

        schema = _OUTPUT_SCHEMAS[task]
        chat_model = resolve_chat_model(self._model)

        def invoke() -> list[Any]:
            if chat_model is None:
                output = self._model.invoke(request)
            else:
                output = chat_model.with_structured_output(
                    schema=schema,
                    method="json_schema",
                ).invoke(request)
            return _extract_examples(output)

        return invoke_with_retry(
            invoke,
            self._retry_config,
            extra_retry_exceptions=(GenerationResponseError,),
        )

    def _build_pipeline(self) -> ValidationPipeline:
        """Build the validation pipeline."""

        return ValidationPipeline(
            validator=ExampleValidator(),
            deduplicator=Deduplicator(),
            judge=LLMJudge(
                self._judge_model,
                quality_threshold=self._judge_quality_threshold,
                batch_size=self._judge_batch_size,
                retry_config=self._retry_config,
            ),
            max_topup_attempts=self._max_topup_attempts,
        )
