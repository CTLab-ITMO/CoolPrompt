"""Build a validated TaskSpec and generation context from a user prompt."""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import Field, ValidationError

from coolprompt.spec_generator.models import (
    Example,
    GenerationContext,
    StrictModel,
    TaskSpec,
    TaskSpecDraft,
)
from coolprompt.spec_generator.utils.model_utils import parse_structured
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    PROBLEM_DESCRIPTION_CLASSIFICATION_TEMPLATE,
    PROBLEM_DESCRIPTION_GENERATION_TEMPLATE,
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


class _ProblemDescription(StrictModel):
    """Structured response for task-description inference."""

    description: str = Field(min_length=1)


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
    """Render trusted examples as JSON."""

    return json.dumps(
        [{"input": example.input, "output": example.output} for example in examples],
        ensure_ascii=False,
        indent=2,
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
        updates.get("task") not in (None, Task.CLASSIFICATION)
        and "labels" not in updates
    ):
        updates["labels"] = None

    return TaskSpec.model_validate(spec.model_dump() | updates)


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
        """Initialize specification inference and optional dataset detection."""

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
        """Build the context used for synthetic generation."""

        dataset = dataset_name or (
            self._detect_dataset(prompt) if detect_dataset else None
        )

        seed_examples, from_dataset = self._resolve_examples(examples, dataset)

        request = _build_request(prompt, seed_examples, dataset, draft)
        spec = self._invoke(request, draft)

        dataset = self._validate_dataset_match(spec, dataset)

        if from_dataset and dataset is None:
            seed_examples = ()

        logger.info("GenerationContext ready: task=%r, dataset=%r", spec.task, dataset)

        return GenerationContext(
            spec=spec,
            dataset_name=dataset,
            seed_examples=seed_examples,
        )

    @staticmethod
    def _resolve_examples(
        examples: Sequence[tuple[str, str] | Example] | None,
        dataset_name: str | None,
    ) -> tuple[tuple[Example, ...], bool]:
        """Resolve user-provided or dataset reference examples."""

        if examples is not None:
            resolved = tuple(
                (
                    item
                    if isinstance(item, Example)
                    else Example(input=item[0], output=item[1])
                )
                for item in examples
            )
            return resolved, False

        resolved = tuple(
            Example(input=item.input, output=item.target)
            for item in DATASET_EXAMPLES.get(dataset_name, ())
        )

        return resolved, bool(resolved)

    @staticmethod
    def _validate_dataset_match(
        spec: TaskSpec,
        dataset_name: str | None,
    ) -> str | None:
        """Return the dataset name when it matches the task specification."""

        if not dataset_name:
            return None

        expected = DATASET_LABEL_SETS.get(dataset_name)
        if expected is None:
            return dataset_name

        if spec.task != Task.CLASSIFICATION or not spec.labels:
            logger.info(
                "Ignoring dataset %r: classification task expected.",
                dataset_name,
            )
            return None

        labels = {label.strip().casefold() for label in spec.labels}
        expected_labels = {label.strip().casefold() for label in expected}

        if labels == expected_labels:
            return dataset_name

        logger.info(
            "Ignoring dataset %r: labels %r do not match %r.",
            dataset_name,
            spec.labels,
            sorted(expected_labels),
        )
        return None

    def _invoke(
        self,
        request: str,
        draft: TaskSpecDraft | None = None,
    ) -> TaskSpec:
        """Invoke the specification model with retry handling."""

        def attempt() -> TaskSpec:
            try:
                spec = parse_structured(
                    self._spec_model,
                    request,
                    TaskSpec,
                    error_cls=SpecResponseError,
                    label="Specification",
                )
                return _apply_draft(spec, draft)
            except ValidationError as exc:
                raise SpecResponseError(
                    "Specification failed validation after applying user overrides."
                ) from exc

        return invoke_with_retry(
            attempt,
            self._retry_config,
            extra_retry_exceptions=(SpecResponseError,),
        )

    def _detect_dataset(self, prompt: str) -> str | None:
        """Detect a reference dataset from the prompt."""

        try:
            detection = self._detector.detect_task_area(prompt)
        except Exception as exc:
            logger.warning("Dataset detection failed: %s", exc)
            return None

        dataset = TASK_AREA_TO_DATASET.get(detection.task_area)

        if dataset:
            logger.info(
                "Detected dataset %r from task area %r (confidence=%.2f).",
                dataset,
                detection.task_area,
                detection.confidence,
            )

        return dataset


def generate_problem_description(
    model: BaseLanguageModel,
    prompt: str,
    *,
    task: Task,
    examples: Sequence[tuple[str, str] | Example] | None = None,
    labels: tuple[str, ...] | None = None,
    retry_config: RetryConfig | None = None,
) -> str:
    """Infer one task-description sentence."""

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    if labels is not None and task != Task.CLASSIFICATION:
        raise ValueError("labels are only valid for classification tasks")

    normalized_examples = tuple(
        item if isinstance(item, Example) else Example(input=item[0], output=item[1])
        for item in (examples or ())
    )

    template = (
        PROBLEM_DESCRIPTION_CLASSIFICATION_TEMPLATE
        if task == Task.CLASSIFICATION
        else PROBLEM_DESCRIPTION_GENERATION_TEMPLATE
    )

    request = template.format(
        prompt=prompt,
        labels=json.dumps(labels, ensure_ascii=False) if labels else "None",
        examples=(
            _render_examples(normalized_examples) if normalized_examples else "None"
        ),
    )

    def invoke() -> str:
        response = parse_structured(
            model,
            request,
            _ProblemDescription,
            error_cls=SpecResponseError,
            label="Problem-description",
        )
        return response.description

    return invoke_with_retry(
        invoke,
        retry_config or RetryConfig(),
        extra_retry_exceptions=(SpecResponseError,),
    )
