"""Task-distribution models and deterministic coverage helpers."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from coolprompt.spec_generator.models import Example, StrictModel, TaskSpec
from coolprompt.spec_generator.utils.model_utils import parse_structured
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.distribution_prompts import (
    DISTRIBUTION_REQUEST_TEMPLATE,
)


class AxisStrategy(str, Enum):
    """Coverage policy for one task axis."""

    BALANCED = "balanced"
    TARGET_PROPORTIONS = "target_proportions"


Constraint = tuple[str, str, str]


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

    def unrecord(self, axis_tags: Mapping[str, str]) -> None:
        """Revert record() without partially changing the state on error."""
        for axis, value in axis_tags.items():
            if self.axis_counts.get(axis, {}).get(value, 0) == 0:
                raise ValueError(f"Cannot unrecord missing axis value: {axis}={value}")

        for axis, value in axis_tags.items():
            counts = self.axis_counts[axis]
            counts[value] -= 1
            if counts[value] == 0:
                del counts[value]
            if not counts:
                del self.axis_counts[axis]


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
        key=lambda i: (target_counts[i] - exact_counts[i], i),
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

        inferred = invoke_with_retry(
            lambda: self._invoke_once(_distribution_request(prompt, spec, examples)),
            self._retry_config,
            extra_retry_exceptions=(DistributionResponseError,),
        )

        label_axis = _label_axis(spec)
        deterministic_axes = [] if label_axis is None else [label_axis]

        reserved_axis_keys = {
            _canonical_axis_key(axis.name) for axis in deterministic_axes
        }

        max_inferred_axes = 5 - len(deterministic_axes)

        inferred_axes = [
            axis
            for axis in inferred.axes
            if _canonical_axis_key(axis.name) not in reserved_axis_keys
        ][:max_inferred_axes]

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

        return parse_structured(
            self._model,
            request,
            TaskDistribution,
            error_cls=DistributionResponseError,
            label="TaskDistribution",
        )


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

    _set_label_axis(result, distribution.axis("label"), output=output, spec=spec)

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


def coverage_gaps(
    distribution: TaskDistribution,
    state: GenerationState,
    total_target: int,
) -> tuple[list[dict], list[dict]]:
    """Return underrepresented and overrepresented distribution values."""

    if total_target <= 0:
        return [], []

    quotas = axis_quotas(distribution, total_target)

    under: list[dict] = []
    over: list[dict] = []

    for axis in distribution.axes:
        counts = state.axis_counts.get(axis.name, {})
        observed = sum(counts.values())

        for value in axis.values:
            actual = counts.get(value.id, 0)
            desired, target_share = quotas[axis.name][value.id]

            if actual < desired:
                under.append(_axis_entry(axis, value, gap=desired - actual))

            if observed:
                share = actual / observed
                if (axis.strategy == AxisStrategy.BALANCED and actual > desired) or (
                    axis.strategy == AxisStrategy.TARGET_PROPORTIONS
                    and share > target_share
                ):
                    over.append(_axis_entry(axis, value, share=share))

    return under, over


def axis_quotas(
    distribution: TaskDistribution,
    total_target: int,
) -> dict[str, dict[str, tuple[int, float]]]:
    """Return target counts and target shares for distribution values."""

    if total_target <= 0:
        raise ValueError("total_target must be positive")

    quotas: dict[str, dict[str, tuple[int, float]]] = {}

    for axis in distribution.axes:
        if axis.strategy == AxisStrategy.BALANCED:
            base, remainder = divmod(total_target, len(axis.values))
            targets = {
                value.id: base + int(index < remainder)
                for index, value in enumerate(axis.values)
            }
        else:
            targets = _target_counts(axis, total_target)

        quotas[axis.name] = {
            value.id: (
                targets[value.id],
                targets[value.id] / total_target,
            )
            for value in axis.values
        }

    return quotas


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


def _allocate_quotas(gaps: list[int], capacity: int) -> list[int]:
    """Allocate slots proportionally to coverage gaps."""

    if capacity <= 0 or not gaps:
        return [0] * len(gaps)

    positive = [max(0, gap) for gap in gaps]
    total = sum(positive)

    if total <= capacity:
        return positive

    raw = [gap * capacity / total for gap in positive]
    quotas = [min(gap, int(value)) for gap, value in zip(positive, raw)]

    leftover = capacity - sum(quotas)

    by_remainder = sorted(
        range(len(positive)),
        key=lambda i: (raw[i] - quotas[i], positive[i], -i),
        reverse=True,
    )

    for i in by_remainder:
        if leftover <= 0:
            break
        if quotas[i] >= positive[i]:
            continue

        quotas[i] += 1
        leftover -= 1

    return quotas


def _allocate_proportional(weights: list[int], capacity: int) -> list[int]:
    """Allocate capacity proportionally without capping quotas at gap sizes."""
    positive = [max(0, weight) for weight in weights]
    total = sum(positive)

    if capacity <= 0 or total <= 0:
        return [0] * len(weights)

    quotas = [weight * capacity // total for weight in positive]
    remainders = [weight * capacity % total for weight in positive]
    leftover = capacity - sum(quotas)

    order = sorted(
        range(len(positive)),
        key=lambda i: (remainders[i], positive[i], -i),
        reverse=True,
    )

    for i in order[:leftover]:
        quotas[i] += 1

    return quotas


def trim_indices(
    tags: Sequence[Mapping[str, str]],
    excess: int,
    quotas: Mapping[str, Mapping[str, tuple[int, float]]],
    axis_counts: Mapping[str, Mapping[str, int]],
    total_target: int,
) -> set[int]:
    """Minimize final quota deficits, then excess shares, when trimming."""

    if excess <= 0:
        return set()

    if excess > len(tags):
        raise ValueError("excess cannot exceed the number of examples")

    if total_target != len(tags) - excess:
        raise ValueError("total_target must equal the number of retained examples")

    values = [
        (axis, value, desired, target_share)
        for axis, axis_values in quotas.items()
        for value, (desired, target_share) in axis_values.items()
    ]
    n, m = len(tags), len(values)
    matrix = lil_matrix((1 + 2 * m, n + 2 * m), dtype=float)
    matrix[0, :n] = 1
    lower = np.full(1 + 2 * m, -np.inf)
    upper = np.full(1 + 2 * m, np.inf)
    lower[0] = upper[0] = total_target

    for j, (axis, value, desired, target_share) in enumerate(values):
        matching = [i for i, item in enumerate(tags) if item.get(axis) == value]
        if axis_counts.get(axis, {}).get(value, 0) != len(matching):
            raise ValueError(
                f"Coverage state differs from accepted tags: {axis}={value}"
            )
        matrix[1 + 2 * j, matching] = 1
        matrix[1 + 2 * j + 1, matching] = 1
        matrix[1 + 2 * j, n + j] = 1
        matrix[1 + 2 * j + 1, n + m + j] = -1
        lower[1 + 2 * j] = desired
        upper[1 + 2 * j + 1] = target_share * total_target

    objective = np.zeros(n + 2 * m)
    objective[n : n + m] = m * total_target + 1
    objective[n + m :] = 1

    bounds = Bounds(
        np.zeros(n + 2 * m),
        np.concatenate((np.ones(n), np.full(2 * m, total_target))),
    )

    result = milp(
        objective,
        integrality=np.r_[np.ones(n), np.zeros(2 * m)],
        bounds=bounds,
        constraints=[LinearConstraint(matrix.tocsr(), lower, upper)],
    )

    if not result.success or result.x is None:
        raise RuntimeError("Unable to trim examples while preserving coverage.")

    return {i for i, selected in enumerate(result.x[:n]) if selected < 0.5}


def _slots_to_targets(
    slots: list[tuple[Constraint, ...]],
) -> list[dict[str, Any]]:
    """Group identical slots into generation targets."""
    counts: Counter[tuple[Constraint, ...]] = Counter(slots)

    return [
        _target(
            count,
            constraints=[
                {
                    "axis": axis,
                    "value_id": value_id,
                    "description": description,
                }
                for axis, value_id, description in constraints
            ],
        )
        for constraints, count in counts.most_common()
    ]


def build_generation_targets(
    distribution: TaskDistribution,
    state: GenerationState,
    *,
    batch_size: int,
    remaining_budget: int,
    total_target: int,
    rng: random.Random | None = None,
    mix_axes: bool = True,
    post_target: bool = False,
    gap_oversample_factor: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build generation targets, using targeted oversampling for post-target gaps."""

    batch_slots = min(batch_size, remaining_budget)

    if batch_slots <= 0:
        return [], []

    under, over = coverage_gaps(distribution, state, total_target)

    gaps = [item for item in under if int(item["gap"]) > 0]

    if post_target:
        total_gap = sum(int(item["gap"]) for item in gaps)

        if total_gap <= 0:
            return [], over

        capacity = min(batch_slots, math.ceil(total_gap * gap_oversample_factor))
        quotas = _allocate_proportional([int(item["gap"]) for item in gaps], capacity)

    else:
        if not gaps:
            return [_target(batch_slots)], over

        quotas = _allocate_quotas([int(item["gap"]) for item in gaps], batch_slots)

    slots: list[tuple[Constraint, ...]] = [
        ((str(item["axis"]), str(item["value_id"]), str(item["description"])),)
        for item, quota in zip(gaps, quotas)
        for _ in range(quota)
    ]

    if not post_target:
        slots.extend([()] * (batch_slots - len(slots)))

    if mix_axes:
        (rng or random.Random()).shuffle(slots)

    return _slots_to_targets(slots), over
