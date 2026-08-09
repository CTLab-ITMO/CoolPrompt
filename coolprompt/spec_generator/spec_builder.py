"""Build a validated TaskSpec and generation context from a user prompt."""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import ValidationError

from coolprompt.spec_generator.models import (
    Example,
    GenerationContext,
    TaskSpec,
    TaskSpecDraft,
)
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    SPEC_FROM_PROMPT_AND_EXAMPLES_TEMPLATE,
    SPEC_FROM_PROMPT_TEMPLATE,
)
from coolprompt.utils.task_areas import (
    DATASET_EXAMPLES,
    DATASET_LABEL_SETS,
    TASK_AREA_TO_DATASET,
)


class SpecResponseError(ValueError):
    """Raised when the specification model returns an invalid response."""


def _render_draft(draft: TaskSpecDraft | None) -> str:
    """Render explicit user overrides for the specification model."""

    if draft is None or draft.is_empty:
        return ""

    payload = json.dumps(
        draft.model_dump(
            exclude_unset=True,
            exclude_none=True,
            mode="json",
        ),
        ensure_ascii=False,
        indent=2,
    )
    return f"\n\nUser-provided overrides. Respect them exactly:\n{payload}"


def _render_examples(examples: Sequence[Example]) -> str:
    """Render trusted examples as escaped XML."""

    return "\n".join(
        f'<example index="{index}">\n'
        f"<input>{escape(example.input)}</input>\n"
        f"<output>{escape(example.output)}</output>\n"
        "</example>"
        for index, example in enumerate(examples, start=1)
    )


def _build_request(
        prompt: str,
        examples: Sequence[Example],
        dataset_name: str | None,
        draft: TaskSpecDraft | None,
) -> str:
    """Build the TaskSpec inference prompt."""

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    dataset_context = (
        f"Detected reference dataset: {dataset_name}. "
        "Use it only as supporting context."
        if dataset_name
        else ""
    )

    values = {
        "prompt": f"{prompt}{_render_draft(draft)}",
        "dataset_context": dataset_context,
    }

    if examples:
        return SPEC_FROM_PROMPT_AND_EXAMPLES_TEMPLATE.format(
            **values,
            examples=_render_examples(examples),
        )

    return SPEC_FROM_PROMPT_TEMPLATE.format(**values)


def _apply_draft(
        spec: TaskSpec,
        draft: TaskSpecDraft | None,
) -> TaskSpec:
    """Apply explicit user overrides and revalidate the specification."""

    if draft is None or draft.is_empty:
        return spec

    updates = draft.overrides()

    if (
            "task" in updates
            and updates["task"] != Task.CLASSIFICATION
            and "labels" not in updates
    ):
        updates["labels"] = None

    return TaskSpec.model_validate(spec.model_dump() | updates)


def _parse_spec(output: Any) -> TaskSpec:
    """Convert a model response into a validated TaskSpec."""

    if isinstance(output, TaskSpec):
        return output

    if isinstance(output, AIMessage):
        output = output.content

    if isinstance(output, str):
        output = extract_json(output)

    if not isinstance(output, dict):
        raise TypeError(f"Unexpected specification response type: {type(output)!r}")

    return TaskSpec.model_validate(output)


class SpecBuilder:
    """Infer a complete TaskSpec from a natural-language prompt."""

    def __init__(
            self,
            model: BaseLanguageModel,
            detector_confidence_threshold: float = 0.7,
            retry_config: RetryConfig | None = None,
            *,
            task_spec_model: BaseLanguageModel | None = None,
    ) -> None:
        self._spec_model = task_spec_model or model
        self._retry_config = retry_config or RetryConfig()
        self._detector = TaskDetector(
            model,
            confidence_threshold=detector_confidence_threshold,
        )

    def build(
            self,
            prompt: str,
            examples: Sequence[tuple[str, str] | Example] | None = None,
            draft: TaskSpecDraft | None = None,
            *,
            detect_dataset: bool = False,
            dataset_name: str | None = None,
    ) -> GenerationContext:
        """Build the immutable context used for synthetic generation."""

        detected_dataset = dataset_name
        if detected_dataset is None and detect_dataset:
            detected_dataset = self._detect_dataset(prompt)

        seed_examples, from_dataset = self._resolve_examples(
            examples,
            detected_dataset,
        )

        spec = _apply_draft(
            self._invoke(
                _build_request(
                    prompt,
                    seed_examples,
                    detected_dataset,
                    draft,
                )
            ),
            draft,
        )

        validated_dataset = self._validate_dataset_match(
            spec,
            detected_dataset,
        )

        if from_dataset and validated_dataset is None:
            seed_examples = ()

        logger.info(
            "GenerationContext ready: task=%r, corner_cases=%d, dataset=%r",
            spec.task,
            len(spec.corner_cases),
            validated_dataset,
        )

        return GenerationContext(
            spec=spec,
            dataset_name=validated_dataset,
            seed_examples=seed_examples,
        )

    @staticmethod
    def _resolve_examples(
            examples: Sequence[tuple[str, str] | Example] | None,
            dataset_name: str | None,
    ) -> tuple[tuple[Example, ...], bool]:
        """Resolve user-provided or dataset reference examples.

        Args:
            examples: Optional user-provided input-output examples.
            dataset_name: Detected reference dataset name.

        Returns:
            A tuple containing resolved examples and whether they came from
            the reference dataset.
        """

        if examples is not None:
            return (
                tuple(
                    item
                    if isinstance(item, Example)
                    else Example(input=item[0], output=item[1])
                    for item in examples
                ),
                False,
            )

        dataset_examples = (
            DATASET_EXAMPLES.get(dataset_name, ())
            if dataset_name
            else ()
        )

        return (
            tuple(
                Example(input=item.input, output=item.target)
                for item in dataset_examples
            ),
            bool(dataset_examples),
        )

    @staticmethod
    def _validate_dataset_match(
            spec: TaskSpec,
            dataset_name: str | None,
    ) -> str | None:
        """Validate that the detected dataset matches the TaskSpec.

        Args:
            spec (TaskSpec): Validated task specification.
            dataset_name (str | None): Detected dataset name.

        Returns:
            str | None: Dataset name when compatible, otherwise None.
        """

        if not dataset_name:
            return None

        expected_labels = DATASET_LABEL_SETS.get(dataset_name)
        if expected_labels is None:
            return dataset_name

        if spec.task != Task.CLASSIFICATION or not spec.labels:
            logger.info("Ignoring dataset %r: classification task expected.", dataset_name)
            return None

        actual = {
            label.strip().casefold()
            for label in spec.labels
        }
        expected = {
            label.strip().casefold()
            for label in expected_labels
        }

        if actual == expected:
            return dataset_name

        logger.info(
            "Ignoring dataset %r: labels %r do not match %r.",
            dataset_name,
            spec.labels,
            sorted(expected_labels),
        )
        return None

    def _invoke(self, request: str) -> TaskSpec:
        """Invoke the specification model with retry handling."""

        return invoke_with_retry(
            lambda: self._invoke_once(request),
            self._retry_config,
            extra_retry_exceptions=(SpecResponseError,),
        )

    def _invoke_once(self, request: str) -> TaskSpec:
        """Invoke and parse one specification-model response."""

        chat_model = resolve_chat_model(self._spec_model)

        try:
            output = (
                self._spec_model.invoke(request)
                if chat_model is None
                else chat_model.with_structured_output(
                    schema=TaskSpec,
                    method="json_schema",
                ).invoke(request)
            )
            return _parse_spec(output)

        except ValidationError as exc:
            raise SpecResponseError("Specification response failed validation.") from exc
        except (TypeError, ValueError) as exc:
            raise SpecResponseError("Specification response could not be parsed.") from exc

    def _detect_dataset(self, prompt: str) -> str | None:
        """Detect a reference dataset from the user prompt."""

        try:
            detection = self._detector.detect_task_area(prompt)
            if detection.task_area is None:
                return None

            dataset_name = TASK_AREA_TO_DATASET.get(detection.task_area)
            if dataset_name is None:
                logger.info("No dataset mapping for task area %r.", detection.task_area)
                return None

            logger.info(
                "Detected dataset %r from task area %r (confidence=%.2f).",
                dataset_name,
                detection.task_area,
                detection.confidence,
            )
            return dataset_name

        except Exception as exc:
            logger.warning("Dataset detection failed: %s", exc)
            return None
