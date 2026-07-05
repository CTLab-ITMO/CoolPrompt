from __future__ import annotations

import random
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.data_generator.pydantic_formatters import (
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskStructuredOutputSchema,
)
from coolprompt.spec_generator.data_spec import DataSpec
from coolprompt.spec_generator.request_builder import RequestBuilder
from coolprompt.spec_generator.schema import GenerationResult, TaskSpec
from coolprompt.spec_generator.spec_builder import SpecBuilder
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry_config import ValidationConfig
from coolprompt.spec_generator.utils.retry_utils import (
    RetryConfig,
    invoke_with_retry,
)
from coolprompt.spec_generator.validation.example_models import ExampleBase
from coolprompt.spec_generator.validation.format_validator import (
    Deduplicator,
    FormatValidator,
)
from coolprompt.spec_generator.validation.judge import LLMJudge
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json

_OUTPUT_SCHEMAS: dict[Task, type[BaseModel]] = {
    Task.CLASSIFICATION: ClassificationTaskStructuredOutputSchema,
    Task.GENERATION: GenerationTaskStructuredOutputSchema,
}


def _split(total: int, corner_ratio: float) -> tuple[int, int]:
    n_corner = int(total * corner_ratio)
    return total - n_corner, n_corner


def _batches(total: int, batch_size: int) -> list[int]:
    return [
        min(batch_size, total - start)
        for start in range(0, total, batch_size)
    ]


def _validate_args(
        num_samples: int,
        corner_ratio: float,
        batch_size: int,
) -> None:
    if not 1 <= num_samples <= 100:
        raise ValueError(f"num_samples must be between 1 and 100, got {num_samples}.")

    if not 0.0 <= corner_ratio <= 1.0:
        raise ValueError(
            f"corner_ratio must be between 0.0 and 1.0, "
            f"got {corner_ratio}."
        )

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")


def _extract_examples(
        payload: Any,
        *,
        source: str,
) -> list[Any]:
    try:
        if isinstance(payload, AIMessage):
            payload = extract_json(payload.content)
        elif isinstance(payload, str):
            payload = extract_json(payload)

        if isinstance(payload, BaseModel):
            examples = getattr(payload, "examples", None)
        elif isinstance(payload, dict):
            examples = payload.get("examples")
        else:
            logger.warning(
                "Unexpected %s response type: %r. "
                "Treating batch as empty.",
                source,
                type(payload),
            )
            return []

        if not isinstance(examples, list):
            logger.warning("%s response has no valid 'examples' list. "
                           "Treating batch as empty.", source)
            return []

        return examples

    except Exception as exc:
        logger.warning("Failed to parse %s response: %s. "
                       "Treating batch as empty.", source, exc)
        return []


class SyntheticDataGenerator:
    def __init__(
            self,
            model: BaseLanguageModel,
            detector_confidence_threshold: float = 0.7,
            validation_config: ValidationConfig | None = None,
            retry_config: RetryConfig | None = None,
    ) -> None:
        self._model = model
        self._validation_config = (
            validation_config
            if validation_config is not None
            else ValidationConfig()
        )
        self._retry_config = (
            retry_config
            if retry_config is not None
            else RetryConfig()
        )

        self._spec_builder = SpecBuilder(
            model,
            detector_confidence_threshold,
            retry_config=self._retry_config,
        )
        self._request_builder = RequestBuilder()

    def build_spec(
            self,
            prompt: str,
            *,
            user_spec: DataSpec | None = None,
            examples: list[tuple[str, str]] | None = None,
    ) -> TaskSpec:
        return self._build_spec(
            prompt=prompt,
            user_spec=user_spec,
            examples=examples,
        )

    def generate(
            self,
            prompt: str,
            task: Task,
            *,
            spec: TaskSpec | None = None,
            user_spec: DataSpec | None = None,
            examples: list[tuple[str, str]] | None = None,
            num_samples: int = 8,
            batch_size: int = 15,
            corner_ratio: float = 0.4,
            validation: bool = False,
    ) -> GenerationResult:
        _validate_args(
            num_samples=num_samples,
            corner_ratio=corner_ratio,
            batch_size=batch_size,
        )

        if task not in _OUTPUT_SCHEMAS:
            supported = ", ".join(
                supported_task.value
                for supported_task in _OUTPUT_SCHEMAS
            )
            raise ValueError(
                f"Unsupported generation task {task!r}. "
                f"Supported tasks: {supported}."
            )

        if spec is None:
            spec = self._build_spec(
                prompt=prompt,
                user_spec=user_spec,
                examples=examples,
            )

        return self._spec_generate(
            spec=spec,
            task=task,
            num_samples=num_samples,
            corner_ratio=corner_ratio,
            batch_size=batch_size,
            validation=validation,
        )

    def _build_spec(
            self,
            prompt: str,
            user_spec: DataSpec | None,
            examples: list[tuple[str, str]] | None,
    ) -> TaskSpec:
        return self._spec_builder.build(
            prompt=prompt,
            examples=examples,
            user_spec=user_spec,
            detect_dataset=False,
        )

    def _spec_generate(
            self,
            spec: TaskSpec,
            task: Task,
            num_samples: int,
            corner_ratio: float,
            batch_size: int,
            validation: bool,
    ) -> GenerationResult:
        n_regular, n_corner = _split(
            total=num_samples,
            corner_ratio=corner_ratio,
        )

        if validation:
            generated = self._generate_validated(
                spec=spec,
                task=task,
                n_regular=n_regular,
                n_corner=n_corner,
                batch_size=batch_size,
            )

            inputs = [example.input for example in generated]
            outputs = [example.output for example in generated]

        else:
            generated = self._generate_unvalidated(
                spec=spec,
                task=task,
                n_regular=n_regular,
                n_corner=n_corner,
                batch_size=batch_size,
            )

            unpacked = [
                self._unpack(example)
                for example in generated
            ]

            if unpacked:
                inputs_tuple, outputs_tuple = zip(*unpacked)
                inputs = list(inputs_tuple)
                outputs = list(outputs_tuple)
            else:
                inputs = []
                outputs = []

        if len(generated) < num_samples:
            logger.warning("Generated fewer examples than requested: "
                           "requested=%d, got=%d.", num_samples, len(generated))

        return GenerationResult(
            dataset=inputs,
            target=outputs,
            spec=spec,
            description=spec.task_summary,
        )

    def _generate_validated(
            self,
            spec: TaskSpec,
            task: Task,
            n_regular: int,
            n_corner: int,
            batch_size: int,
    ) -> list[ExampleBase]:
        pipeline = self._build_pipeline()
        total_target = n_regular + n_corner

        corner = self._run_validated_group(
            pipeline=pipeline,
            spec=spec,
            task=task,
            target_n=n_corner,
            batch_size=batch_size,
            is_corner=True,
        )

        regular_target = total_target - len(corner)

        self._log_corner_reallocation(
            requested_corner=n_corner,
            actual_corner=len(corner),
            original_regular=n_regular,
            regular_target=regular_target,
        )

        regular = self._run_validated_group(
            pipeline=pipeline,
            spec=spec,
            task=task,
            target_n=regular_target,
            batch_size=batch_size,
            is_corner=False,
        )

        return (corner + regular)[:total_target]

    def _generate_unvalidated(
            self,
            spec: TaskSpec,
            task: Task,
            n_regular: int,
            n_corner: int,
            batch_size: int,
    ) -> list[Any]:
        total_target = n_regular + n_corner

        corner = self._generate_group(
            spec=spec,
            task=task,
            n=n_corner,
            batch_size=batch_size,
            is_corner=True,
        )[:n_corner]

        regular_target = total_target - len(corner)

        self._log_corner_reallocation(
            requested_corner=n_corner,
            actual_corner=len(corner),
            original_regular=n_regular,
            regular_target=regular_target,
        )

        regular = self._generate_group(
            spec=spec,
            task=task,
            n=regular_target,
            batch_size=batch_size,
            is_corner=False,
        )[:regular_target]

        return (corner + regular)[:total_target]

    def _run_validated_group(
            self,
            pipeline: ValidationPipeline,
            spec: TaskSpec,
            task: Task,
            target_n: int,
            batch_size: int,
            *,
            is_corner: bool,
    ) -> list[ExampleBase]:
        if target_n <= 0:
            return []

        if is_corner and not self._can_generate_corner(spec):
            logger.warning(
                "No corner-case source available; "
                "skipping corner generation."
            )
            return []

        return pipeline.run(
            raw_batch_producer=lambda remaining: self._generate_group(
                spec=spec,
                task=task,
                n=remaining,
                batch_size=batch_size,
                is_corner=is_corner,
            ),
            spec=spec,
            task=task,
            target_n=target_n,
            is_corner=is_corner,
        )

    def _generate_group(
            self,
            spec: TaskSpec,
            task: Task,
            n: int,
            batch_size: int,
            *,
            is_corner: bool,
    ) -> list[Any]:
        if n <= 0:
            return []

        group_name = "corner" if is_corner else "regular"

        logger.info("Generating %d %s samples in batches of %d.", n, group_name, batch_size)

        examples: list[Any] = []

        for batch in _batches(n, batch_size):
            request = self._build_request(
                spec=spec,
                task=task,
                n=batch,
                is_corner=is_corner,
            )

            if request is None:
                logger.warning(
                    "No corner cases in spec; "
                    "stopping corner generation."
                )
                break

            examples.extend(
                self._call_model(
                    request=request,
                    task=task,
                )
            )

        return examples

    def _build_pipeline(self) -> ValidationPipeline:
        return ValidationPipeline(
            format_validator=FormatValidator(),
            deduplicator=Deduplicator(),
            judge=LLMJudge(
                self._model,
                self._validation_config,
                self._retry_config,
            ),
            config=self._validation_config,
        )

    def _build_request(
            self,
            spec: TaskSpec,
            task: Task,
            n: int,
            *,
            is_corner: bool,
    ) -> str | None:
        if is_corner:
            return self._build_corner_request(
                spec=spec,
                task=task,
                n=n,
            )

        return self._build_regular_request(
            spec=spec,
            task=task,
            n=n,
        )

    def _build_regular_request(
            self,
            spec: TaskSpec,
            task: Task,
            n: int,
    ) -> str:
        if spec.matched_dataset:
            request = self._request_builder.dataset_regular(
                spec,
                spec.matched_dataset,
                n,
            )

            if request is not None:
                return request

        return self._request_builder.regular(
            spec,
            task,
            n,
        )

    def _build_corner_request(
            self,
            spec: TaskSpec,
            task: Task,
            n: int,
    ) -> str | None:
        if spec.matched_dataset:
            request = self._request_builder.dataset_corner(
                spec,
                spec.matched_dataset,
                n,
            )

            if request:
                return request

        if not spec.corner_cases:
            return None

        patterns = random.sample(
            spec.corner_cases,
            min(len(spec.corner_cases), n),
        )

        return self._request_builder.corner(spec, task, patterns, n)

    @staticmethod
    def _can_generate_corner(spec: TaskSpec) -> bool:
        return bool(
            spec.corner_cases
            or spec.matched_dataset
        )

    @staticmethod
    def _log_corner_reallocation(
            requested_corner: int,
            actual_corner: int,
            original_regular: int,
            regular_target: int,
    ) -> None:
        shortfall = requested_corner - actual_corner

        if shortfall <= 0:
            return

        logger.info(
            "Corner generation produced %d/%d examples; "
            "reallocating shortfall=%d to regular target (%d -> %d).",
            actual_corner,
            requested_corner,
            shortfall,
            original_regular,
            regular_target,
        )

    def _call_model(
            self,
            request: str,
            task: Task,
    ) -> list[Any]:
        schema = _OUTPUT_SCHEMAS[task]
        chat_model = resolve_chat_model(self._model)

        if chat_model is None:
            raw = invoke_with_retry(
                lambda: self._model.invoke(request),
                self._retry_config,
            )

            return _extract_examples(
                raw,
                source="generation",
            )

        output = invoke_with_retry(
            lambda: (
                chat_model
                .with_structured_output(
                    schema=schema,
                    method="json_schema",
                )
                .invoke(request)
            ),
            self._retry_config,
        )

        return _extract_examples(
            output,
            source="structured generation",
        )

    @staticmethod
    def _unpack(example: Any) -> tuple[str, str]:
        if isinstance(example, dict):
            return (
                example["input"],
                example["output"],
            )

        return (
            example.input,
            example.output,
        )
