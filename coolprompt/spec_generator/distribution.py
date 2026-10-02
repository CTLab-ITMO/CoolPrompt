"""Task-distribution models and deterministic coverage helpers."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeVar

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from coolprompt.spec_generator.models import Example, StrictModel, TaskSpec
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.distribution_prompts import (
    DISTRIBUTION_REQUEST_TEMPLATE,
)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


class AxisStrategy(str, Enum):
    """Coverage policy for one task axis."""

    BALANCED = "balanced"
    TARGET_PROPORTIONS = "target_proportions"


class AxisValue(StrictModel):
    """One named value on a task-distribution axis."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class TaskAxis(StrictModel):
    """One meaningful variation axis of a task."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    strategy: AxisStrategy = AxisStrategy.BALANCED
    values: tuple[AxisValue, ...]

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: tuple[AxisValue, ...]) -> tuple[AxisValue, ...]:
        """Require at least two uniquely identified values per axis."""

        if len(values) < 2:
            raise ValueError("A task axis must contain at least two values.")

        if len({v.id.casefold() for v in values}) != len(values):
            raise ValueError("Axis value ids must be unique within an axis.")

        return values

    @model_validator(mode="after")
    def validate_strategy(self) -> "TaskAxis":
        """Validate ratios for the selected coverage strategy."""

        ratios = [value.target_ratio for value in self.values]

        if self.strategy == AxisStrategy.BALANCED:
            if any(ratio is not None for ratio in ratios):
                raise ValueError("BALANCED must not define target_ratio.")
        else:
            if any(ratio is None for ratio in ratios):
                raise ValueError(
                    "TARGET_PROPORTIONS requires target_ratio for every value."
                )

            if not 0.95 <= sum(ratio for ratio in ratios if ratio is not None) <= 1.05:
                raise ValueError("target_ratio values must sum approximately to 1.0.")

        return self


def _canonical_axis_key(value: str) -> str:
    """Normalize equivalent axis-name spellings for matching."""

    return " ".join(
        value.strip().casefold().replace("_", " ").replace("-", " ").split()
    )


class TaskDistribution(StrictModel):
    """Meaningful task-variation axes to cover."""

    axes: tuple[TaskAxis, ...]

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, axes: tuple[TaskAxis, ...]) -> tuple[TaskAxis, ...]:
        """Require one to five axes with unique normalized names."""

        if not 1 <= len(axes) <= 5:
            raise ValueError("TaskDistribution must contain 1-5 axes.")

        if len({_canonical_axis_key(a.name) for a in axes}) != len(axes):
            raise ValueError("Task axis names must be unique.")

        return axes

    def axis(self, name: str) -> TaskAxis | None:
        """Return an axis by its normalized name, if present."""

        key = _canonical_axis_key(name)
        return next((a for a in self.axes if _canonical_axis_key(a.name) == key), None)


class GenerationState(BaseModel):
    """Coverage state for accepted examples in the current generation run."""

    axis_counts: dict[str, dict[str, int]] = Field(default_factory=dict)

    def record(self, axis_tags: Mapping[str, str]) -> None:
        """Increment observed counts for a generated example's axis tags."""

        for axis_name, value_id in axis_tags.items():
            counts = self.axis_counts.setdefault(axis_name, {})
            counts[value_id] = counts.get(value_id, 0) + 1


class TaggedGeneratedExample(BaseModel):
    """Private structured output for distribution-aware generation."""

    input: str = Field(min_length=1)
    output: str
    axis_tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("axis_tags", mode="before")
    @classmethod
    def normalize_axis_tags(cls, value: Any) -> dict[str, str]:
        """Normalize structured-output axis tags into a string mapping."""

        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise ValueError("axis_tags must be a mapping")

        return {str(axis): str(tag) for axis, tag in value.items() if tag is not None}


class TaggedGenerationBatch(BaseModel):
    """Structured batch of generated examples."""

    examples: list[TaggedGeneratedExample]


class DistributionResponseError(ValueError):
    """Raised when TaskDistribution inference returns unusable output."""


def _render_examples(examples: Sequence[Example], *, limit: int = 30) -> str:
    """Render a bounded set of trusted examples as JSON."""

    return (
        json.dumps(
            [{"input": e.input, "output": e.output} for e in examples[:limit]],
            ensure_ascii=False,
            indent=2,
        )
        if examples
        else "None"
    )


def _distribution_request(
    prompt: str,
    spec: TaskSpec,
    examples: Sequence[Example],
) -> str:
    """Build the prompt used to infer non-deterministic coverage axes."""

    labels = list(spec.labels or ())

    label_rule = (
        "A label axis is added deterministically from TaskSpec.labels. "
        "Do not return a label/class axis."
        if spec.task == Task.CLASSIFICATION and labels
        else ""
    )

    payload = {
        "task": spec.task.value,
        "description": spec.description,
        "input_format": spec.input_format,
        "output_format": spec.output_format,
        "requirements": list(spec.requirements),
        "labels": labels or None,
    }

    return DISTRIBUTION_REQUEST_TEMPLATE.format(
        prompt=prompt.strip(),
        payload_json=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        examples=_render_examples(
            examples,
            limit=8,
        ),
        label_rule=label_rule,
    )


def _label_axis(spec: TaskSpec) -> TaskAxis | None:
    """Build a deterministic label axis for classification tasks."""

    if spec.task != Task.CLASSIFICATION or not spec.labels or len(spec.labels) < 2:
        return None

    return TaskAxis(
        name="label",
        description="The required classification label.",
        values=tuple(
            AxisValue(
                id=f"label:{i}",
                description=label,
            )
            for i, label in enumerate(spec.labels)
        ),
    )


def _normalize_axis_ratios(axis: TaskAxis) -> TaskAxis:
    """Normalize rounded target proportions to sum exactly to one."""

    if axis.strategy != AxisStrategy.TARGET_PROPORTIONS:
        return axis

    total = sum(value.target_ratio or 0.0 for value in axis.values)

    if total <= 0:
        return axis

    return TaskAxis(
        name=axis.name,
        description=axis.description,
        strategy=axis.strategy,
        values=tuple(
            AxisValue(
                id=value.id,
                description=value.description,
                target_ratio=(value.target_ratio or 0.0) / total,
            )
            for value in axis.values
        ),
    )


def _target_counts(axis: TaskAxis, total_target: int) -> dict[str, int]:
    """Allocate target counts using the largest-remainder method."""

    axis = _normalize_axis_ratios(axis)
    exact_counts = [(value.target_ratio or 0.0) * total_target for value in axis.values]
    target_counts = [math.floor(count) for count in exact_counts]

    remainder_order = sorted(
        range(len(exact_counts)),
        key=lambda index: (
            target_counts[index] - exact_counts[index],
            index,
        ),
    )

    remaining = total_target - sum(target_counts)

    for index in remainder_order[:remaining]:
        target_counts[index] += 1

    return {value.id: count for value, count in zip(axis.values, target_counts)}


class _TaskDistributionBuilder:
    """Infer and validate TaskDistribution once per generate() call."""

    def __init__(
        self,
        model: BaseLanguageModel,
        retry_config: RetryConfig,
    ) -> None:
        """Initialize the builder with a language model and retry policy."""

        self._model = model
        self._retry_config = retry_config

    def build(
        self,
        prompt: str,
        spec: TaskSpec,
        examples: Sequence[Example],
    ) -> TaskDistribution:
        """Infer task-level coverage axes."""

        seed_examples = tuple(examples)

        inferred = invoke_with_retry(
            lambda: self._invoke_once(
                _distribution_request(
                    prompt,
                    spec,
                    seed_examples,
                )
            ),
            self._retry_config,
            extra_retry_exceptions=(DistributionResponseError,),
        )

        deterministic_axes = [axis for axis in (_label_axis(spec),) if axis is not None]

        reserved_axis_keys = {
            _canonical_axis_key(axis.name) for axis in deterministic_axes
        }

        inferred_axes = [
            axis
            for axis in inferred.axes
            if _canonical_axis_key(axis.name) not in reserved_axis_keys
        ][:4]

        return TaskDistribution(
            axes=tuple(
                _normalize_axis_ratios(axis)
                for axis in deterministic_axes + inferred_axes
            )
        )

    def _invoke_once(
        self,
        request: str,
    ) -> TaskDistribution:
        """Invoke the model once and parse a TaskDistribution."""

        return self._invoke_structured(
            request,
            TaskDistribution,
            invalid_type_msg="Unexpected output type",
            validation_msg="TaskDistribution failed validation.",
            parse_msg="TaskDistribution could not be parsed.",
        )

    def _invoke_structured(
        self,
        request: str,
        schema: type[_SchemaT],
        *,
        invalid_type_msg: str,
        validation_msg: str,
        parse_msg: str,
    ) -> _SchemaT:
        """Invoke the model with structured output and validate it."""

        try:
            chat_model = resolve_chat_model(self._model)

            if chat_model is None:
                raw = self._model.invoke(request)
                if isinstance(raw, schema):
                    return raw
                if isinstance(raw, dict):
                    return schema.model_validate(raw)
                content = raw.content if isinstance(raw, AIMessage) else str(raw)

                return schema.model_validate(extract_json(content))

            output = chat_model.with_structured_output(
                schema=schema,
                method="json_schema",
            ).invoke(request)

            if isinstance(output, schema):
                return output

            if isinstance(output, dict):
                return schema.model_validate(output)

            if isinstance(output, AIMessage):
                return schema.model_validate(extract_json(output.content))

            raise DistributionResponseError(f"{invalid_type_msg}: {type(output)!r}")

        except DistributionResponseError:
            raise

        except ValidationError as exc:
            raise DistributionResponseError(validation_msg) from exc

        except (TypeError, ValueError) as exc:
            raise DistributionResponseError(parse_msg) from exc


def validate_axis_tags(
    distribution: TaskDistribution,
    raw_tags: Mapping[str, str] | None,
    *,
    output: str | None = None,
    spec: TaskSpec | None = None,
) -> dict[str, str]:
    """Validate generated axis tags."""

    tags = {
        _canonical_axis_key(name): value_id
        for name, value_id in (raw_tags or {}).items()
    }

    result = {
        axis.name: value_id
        for axis in distribution.axes
        if (value_id := tags.get(_canonical_axis_key(axis.name)))
        in {value.id for value in axis.values}
    }

    _set_label_axis(
        result,
        distribution.axis("label"),
        output=output,
        spec=spec,
    )

    return result


def _set_label_axis(
    result: dict[str, str],
    axis: TaskAxis | None,
    *,
    output: str | None = None,
    spec: TaskSpec | None = None,
) -> None:
    """Derive and validate a deterministic classification label."""

    if axis is None:
        return

    result.pop(axis.name, None)

    if output is None or spec is None or not spec.labels:
        return

    value_id = next(
        (
            f"label:{i}"
            for i, label in enumerate(spec.labels)
            if label.strip().casefold() == output.strip().casefold()
        ),
        None,
    )

    if value_id in {value.id for value in axis.values}:
        result[axis.name] = value_id


def _axis_entry(
    axis: TaskAxis,
    value: AxisValue,
    **extra: Any,
) -> dict[str, Any]:
    """Serialize an axis-value pair with optional coverage metadata."""

    return {
        "axis": axis.name,
        "value_id": value.id,
        "description": value.description,
        **extra,
    }


def _desired_and_allowed_share(
    axis: TaskAxis,
    value: AxisValue,
    target_counts: dict[str, int],
    k: int,
    total_target: int,
    balanced_floor_fraction: float,
    balanced_over_fraction: float,
) -> tuple[int, float]:
    """Return desired count and maximum tolerated share for one value."""

    if axis.strategy == AxisStrategy.TARGET_PROPORTIONS:
        return target_counts[value.id], (value.target_ratio or 0) + 0.10

    index = next(i for i, item in enumerate(axis.values) if item.id == value.id)
    allocation = total_target // k + (index < total_target % k)
    desired = math.ceil(allocation * balanced_floor_fraction)

    return desired, balanced_over_fraction / k


def coverage_gaps(
    distribution: TaskDistribution,
    state: GenerationState,
    total_target: int,
    *,
    balanced_floor_fraction: float = 0.70,
    balanced_over_fraction: float = 1.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return under- and overrepresented axis values."""
    if total_target <= 0:
        return [], []

    under: list[dict[str, Any]] = []
    over: list[dict[str, Any]] = []

    for axis in distribution.axes:
        counts = state.axis_counts.get(axis.name, {})
        observed = sum(counts.values())

        targets = (
            _target_counts(axis, total_target)
            if axis.strategy == AxisStrategy.TARGET_PROPORTIONS
            else {}
        )

        for value in axis.values:
            actual = counts.get(value.id, 0)
            desired, allowed = _desired_and_allowed_share(
                axis,
                value,
                targets,
                len(axis.values),
                total_target,
                balanced_floor_fraction,
                balanced_over_fraction,
            )

            if actual < desired:
                under.append(_axis_entry(axis, value, gap=desired - actual))

            if observed and (share := actual / observed) > allowed:
                over.append(_axis_entry(axis, value, share=share))

    under.sort(key=lambda x: (-x["gap"], x["axis"], x["value_id"]))
    over.sort(key=lambda x: (-x["share"], x["axis"], x["value_id"]))

    return under, over


def _target(
    count: int,
    *,
    constraints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build one generation-target instruction."""

    return {
        "count": count,
        "constraints": constraints or [],
    }


def build_generation_targets(
    distribution: TaskDistribution,
    state: GenerationState,
    *,
    batch_size: int,
    remaining_budget: int,
    total_target: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build coverage targets from current gaps."""

    batch_slots = min(batch_size, remaining_budget)

    if batch_slots <= 0:
        return [], []

    under, over = coverage_gaps(distribution, state, total_target)

    if not under:
        return [_target(batch_slots)], over

    queues: dict[str, list[dict[str, Any]]] = {}
    for item in under:
        queue = queues.setdefault(str(item["axis"]), [])
        queue.extend([item] * min(int(item["gap"]), batch_slots))

    selected: list[dict[str, Any]] = []
    while len(selected) < batch_slots and any(queues.values()):
        for queue in queues.values():
            if queue and len(selected) < batch_slots:
                selected.append(queue.pop(0))

    counts = Counter(
        (str(item["axis"]), str(item["value_id"]), str(item["description"]))
        for item in selected
    )
    targets = [
        _target(
            count,
            constraints=[
                {
                    "axis": axis,
                    "value_id": value_id,
                    "description": description,
                }
            ],
        )
        for (axis, value_id, description), count in counts.items()
    ]
    if len(selected) < batch_slots:
        targets.append(_target(batch_slots - len(selected)))
    return targets, over
