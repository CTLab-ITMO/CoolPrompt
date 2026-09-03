"""Task-distribution models and deterministic coverage helpers."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from html import escape
from typing import Any, TypeVar

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from coolprompt.spec_generator.models import Example, StrictModel, TaskSpec
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.distribution_prompts import DISTRIBUTION_REQUEST_TEMPLATE, \
    AXIS_DEDUP_REQUEST_TEMPLATE


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
        if len(values) < 2:
            raise ValueError("A task axis must contain at least two values.")
        if len({v.id.casefold() for v in values}) != len(values):
            raise ValueError("Axis value ids must be unique within an axis.")
        return values

    @model_validator(mode="after")
    def validate_strategy(self) -> "TaskAxis":
        ratios = [v.target_ratio for v in self.values]

        if self.strategy == AxisStrategy.BALANCED:
            if any(r is not None for r in ratios):
                raise ValueError("BALANCED must not define target_ratio.")
            return self

        if any(r is None for r in ratios):
            raise ValueError("TARGET_PROPORTIONS requires target_ratio for every value.")

        total = sum(r for r in ratios if r is not None)
        if not 0.95 <= total <= 1.05:
            raise ValueError("target_ratio values must sum approximately to 1.0.")

        return self


def _canonical_axis_key(value: str) -> str:
    """Normalize equivalent axis-name spellings for matching."""

    return " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())


class TaskDistribution(StrictModel):
    """Meaningful task-variation axes to cover."""

    axes: tuple[TaskAxis, ...]

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, axes: tuple[TaskAxis, ...]) -> tuple[TaskAxis, ...]:
        if not 1 <= len(axes) <= 5:
            raise ValueError("TaskDistribution must contain 1-5 axes.")
        if len({_canonical_axis_key(a.name) for a in axes}) != len(axes):
            raise ValueError("Task axis names must be unique.")
        return axes

    def axis(self, name: str) -> TaskAxis | None:
        canonical_name = _canonical_axis_key(name)
        return next(
            (a for a in self.axes if _canonical_axis_key(a.name) == canonical_name),
            None,
        )


class GenerationState(BaseModel):
    """Coverage state for accepted examples in the current generation run."""

    axis_counts: dict[str, dict[str, int]] = Field(default_factory=dict)

    def record(self, axis_tags: Mapping[str, str]) -> None:
        for axis_name, value_id in axis_tags.items():
            counts = self.axis_counts.setdefault(axis_name, {})
            counts[value_id] = counts.get(value_id, 0) + 1


class TaggedGeneratedExample(BaseModel):
    """Private structured output for distribution-aware generation."""

    input: str = Field(min_length=1)
    output: str = Field(min_length=1)

    references: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative valid outputs. "
            "Return an empty list when no references are available."
        ),
    )

    axis_tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("references", mode="before")
    @classmethod
    def normalize_references(cls, value: Any) -> list[str]:
        """
        LLM structured output may return:
            "references": null

        Internally references must always be represented as a list.
        """

        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, tuple):
            return [str(item) for item in value]

        if isinstance(value, list):
            return [
                str(item)
                for item in value
                if item is not None
            ]

        raise ValueError(
            "references must be a list, string, or null"
        )

    @field_validator("axis_tags", mode="before")
    @classmethod
    def normalize_axis_tags(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise ValueError("axis_tags must be a mapping")

        return {
            str(axis): str(tag)
            for axis, tag in value.items()
            if tag is not None
        }


class TaggedGenerationBatch(BaseModel):
    """Structured batch of generated examples."""

    examples: list[TaggedGeneratedExample]


class DistributionResponseError(ValueError):
    """Raised when TaskDistribution inference returns unusable output."""


class AxisDedupAction(str, Enum):
    """Decision for one inferred task axis."""

    KEEP = "keep"
    DROP = "drop"


class AxisDedupDecision(StrictModel):
    """Semantic deduplication decision for one candidate axis."""

    axis_name: str = Field(min_length=1)
    action: AxisDedupAction
    duplicate_of: str | None = None
    reason: str = Field(min_length=1)


class AxisDedupResponse(StrictModel):
    """Structured response from the semantic axis-deduplication judge."""

    decisions: tuple[AxisDedupDecision, ...]


def _render_examples(examples: Sequence[Example], *, limit: int = 30) -> str:
    if not examples:
        return "None"

    return "\n".join(
        f'<example index="{i}">\n<input>{escape(e.input)}</input>\n'
        f"<output>{escape(e.output)}</output>\n</example>"
        for i, e in enumerate(examples[:limit], start=1)
    )


def _parse_sequence_size(value: str) -> int | None:
    """Return length for list-like serialized inputs, otherwise None."""

    try:
        parsed = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        return None

    return len(parsed) if isinstance(parsed, (list, tuple)) and parsed else None


def _input_size_axis(reference_examples: Sequence[Example]) -> TaskAxis | None:
    """Build an empirical list-input cardinality axis."""

    if len(reference_examples) < 10:
        return None

    sizes = [
        size
        for example in reference_examples
        if (size := _parse_sequence_size(example.input)) is not None
    ]

    if len(sizes) / len(reference_examples) < 0.8:
        return None

    counts = Counter(sizes)
    if not 2 <= len(counts) <= 6:
        return None

    total = sum(counts.values())

    return TaskAxis(
        name="input_size",
        description=(
            "Number of items in the serialized list input. Preserve the empirical "
            "source-data mix rather than collapsing to one input size."
        ),
        strategy=AxisStrategy.TARGET_PROPORTIONS,
        values=tuple(
            AxisValue(
                id=f"size:{size}",
                description=f"Input contains exactly {size} list items/concepts.",
                target_ratio=count / total,
            )
            for size, count in sorted(counts.items())
        ),
    )


def _distribution_request(
    prompt: str,
    spec: TaskSpec,
    seed_examples: Sequence[Example],
    reference_examples: Sequence[Example],
) -> str:
    labels = list(spec.labels or ())

    label_rule = (
        "A label axis is added deterministically from TaskSpec.labels. "
        "Do not return a label/class axis."
        if spec.task == Task.CLASSIFICATION and labels
        else ""
    )

    empirical_rule = (
        "You have enough distribution-reference examples to use TARGET_PROPORTIONS "
        "for axes whose proportions are directly and repeatedly observable in that "
        "sample."
        if len(reference_examples) >= 20
        else (
            "The distribution-reference sample is small. "
            "Use BALANCED; do not infer target proportions."
        )
    )

    payload = {
        "task": spec.task.value,
        "description": spec.description,
        "input_format": spec.input_format,
        "output_format": spec.output_format,
        "requirements": list(spec.requirements),
        "labels": labels or None,
        "corner_cases": list(spec.corner_cases),
    }

    return DISTRIBUTION_REQUEST_TEMPLATE.format(
        prompt=prompt.strip(),
        payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        seed_examples=_render_examples(seed_examples, limit=8),
        reference_examples=_render_examples(reference_examples, limit=30),
        empirical_rule=empirical_rule,
        label_rule=label_rule,
    )


def _axis_payload(axis: TaskAxis) -> dict[str, Any]:
    return {
        "name": axis.name,
        "description": axis.description,
        "values": [{"id": v.id, "description": v.description} for v in axis.values],
    }


def _axis_dedup_request(
    *,
    spec: TaskSpec,
    deterministic_axes: Sequence[TaskAxis],
    inferred_axes: Sequence[TaskAxis],
) -> str:
    """Build the semantic axis-deduplication judge request."""

    payload = {
        "task": spec.task.value,
        "description": spec.description,
        "labels": list(spec.labels or ()),
        "deterministic_axes": [_axis_payload(a) for a in deterministic_axes],
        "candidate_axes": [_axis_payload(a) for a in inferred_axes],
    }

    return AXIS_DEDUP_REQUEST_TEMPLATE.format(
        payload_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _label_axis(spec: TaskSpec) -> TaskAxis | None:
    if spec.task != Task.CLASSIFICATION or not spec.labels:
        return None

    return TaskAxis(
        name="label",
        description="The required classification label.",
        values=tuple(
            AxisValue(id=f"label:{i}", description=label)
            for i, label in enumerate(spec.labels)
        ),
    )


def _normalize_axis_ratios(axis: TaskAxis) -> TaskAxis:
    """Normalize rounded target proportions to sum exactly to one."""

    if axis.strategy != AxisStrategy.TARGET_PROPORTIONS:
        return axis

    total = sum(v.target_ratio or 0.0 for v in axis.values)
    if total <= 0:
        return axis

    return TaskAxis(
        name=axis.name,
        description=axis.description,
        strategy=axis.strategy,
        values=tuple(
            AxisValue(
                id=v.id,
                description=v.description,
                target_ratio=(v.target_ratio or 0.0) / total,
            )
            for v in axis.values
        ),
    )


def _target_counts(axis: TaskAxis, total_target: int) -> dict[str, int]:
    """Allocate TARGET_PROPORTIONS counts with largest remainder.

    Independent ceil() per value can request more than total_target.
    Largest-remainder allocation preserves the ratios while making the desired
    counts sum exactly to the dataset budget.
    """

    raw = [(v.target_ratio or 0.0) * total_target for v in axis.values]
    floors = [math.floor(r) for r in raw]
    remainder = total_target - sum(floors)

    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:remainder]:
        floors[i] += 1

    return {v.id: floors[i] for i, v in enumerate(axis.values)}


class _TaskDistributionBuilder:
    """Infer and validate TaskDistribution once per generate() call."""

    def __init__(self, model: BaseLanguageModel, retry_config: RetryConfig) -> None:
        self._model = model
        self._retry_config = retry_config

    def build(
        self,
        prompt: str,
        spec: TaskSpec,
        examples: Sequence[Example],
        *,
        reference_examples: Sequence[Example] | None = None,
    ) -> TaskDistribution:
        seed_examples = tuple(examples)
        reference = tuple(reference_examples or seed_examples)

        inferred = invoke_with_retry(
            lambda: self._invoke_once(
                _distribution_request(prompt, spec, seed_examples, reference)
            ),
            self._retry_config,
            extra_retry_exceptions=(DistributionResponseError,),
        )

        deterministic_axes = [
            _normalize_axis_ratios(axis)
            for axis in (_label_axis(spec), _input_size_axis(reference))
            if axis is not None
        ]

        reserved_axis_keys = {
            "label", "labels", "class", "classes",
            "input size", "concept count", "concepts count",
            "cardinality", "input length",
        }

        inferred_axes = [
            _normalize_axis_ratios(axis)
            for axis in inferred.axes
            if _canonical_axis_key(axis.name) not in reserved_axis_keys
        ]

        inferred_axes = invoke_with_retry(
            lambda: self._deduplicate_axes(
                spec=spec,
                deterministic_axes=deterministic_axes,
                inferred_axes=inferred_axes,
            ),
            self._retry_config,
            extra_retry_exceptions=(DistributionResponseError,),
        )

        return TaskDistribution(axes=tuple((deterministic_axes + inferred_axes)[:5]))

    def _deduplicate_axes(
        self,
        *,
        spec: TaskSpec,
        deterministic_axes: Sequence[TaskAxis],
        inferred_axes: Sequence[TaskAxis],
    ) -> list[TaskAxis]:
        """Remove inferred axes that semantically duplicate another axis.

        Deterministic axes are authoritative and are never removed.
        """

        if not inferred_axes:
            return []

        request = _axis_dedup_request(
            spec=spec,
            deterministic_axes=deterministic_axes,
            inferred_axes=inferred_axes,
        )

        response = self._invoke_structured(
            request,
            AxisDedupResponse,
            invalid_type_msg="Unexpected axis-dedup output type",
            validation_msg="Axis deduplication response failed validation.",
            parse_msg="Axis deduplication response could not be parsed.",
        )

        decisions = {
            _canonical_axis_key(d.axis_name): d for d in response.decisions
        }

        return [
            axis
            for axis in inferred_axes
            if (d := decisions.get(_canonical_axis_key(axis.name))) is None
            or d.action == AxisDedupAction.KEEP
        ]

    def _invoke_once(self, request: str) -> TaskDistribution:
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
        """Shared structured-output invocation for both LLM call sites.

        The unexpected-output-type case is re-raised as-is (via the explicit
        `except DistributionResponseError: raise` below) so its message isn't
        swallowed by the broader ValueError handler — note that
        DistributionResponseError itself subclasses ValueError.
        """

        try:
            chat_model = resolve_chat_model(self._model)

            if chat_model is None:
                raw = self._model.invoke(request)
                content = raw.content if isinstance(raw, AIMessage) else str(raw)
                return schema.model_validate(extract_json(content))

            output = chat_model.with_structured_output(
                schema=schema, method="json_schema"
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
    input: str | None = None,
    output: str | None = None,
    spec: TaskSpec | None = None,
) -> dict[str, str]:
    """Keep valid assignments and deterministically override observable axes."""

    result: dict[str, str] = {}
    raw_tags = raw_tags or {}

    normalized_raw_tags = {
        _canonical_axis_key(name): value_id for name, value_id in raw_tags.items()
    }

    for axis in distribution.axes:
        allowed = {v.id for v in axis.values}
        value_id = normalized_raw_tags.get(_canonical_axis_key(axis.name))
        if value_id in allowed:
            result[axis.name] = value_id

    if (input_size_axis := distribution.axis("input_size")) is not None and input is not None:
        size = _parse_sequence_size(input)
        value_id = f"size:{size}" if size is not None else None

        if value_id is not None and any(v.id == value_id for v in input_size_axis.values):
            result[input_size_axis.name] = value_id
        else:
            result.pop(input_size_axis.name, None)

    if (
        (label_axis := distribution.axis("label")) is not None
        and output is not None
        and spec is not None
        and spec.labels
    ):
        canonical_output = output.strip().casefold()
        matched = False

        for i, label in enumerate(spec.labels):
            if label.strip().casefold() == canonical_output:
                result[label_axis.name] = f"label:{i}"
                matched = True
                break

        if not matched:
            result.pop(label_axis.name, None)

    return result


def _axis_entry(axis: TaskAxis, value: AxisValue, **extra: Any) -> dict[str, Any]:
    return {"axis": axis.name, "value_id": value.id, "description": value.description, **extra}


def _desired_and_allowed_share(
    axis: TaskAxis,
    value: AxisValue,
    target_counts: dict[str, int],
    k: int,
    total_target: int,
    balanced_floor_fraction: float,
    balanced_over_fraction: float,
) -> tuple[int, float]:
    if axis.strategy == AxisStrategy.TARGET_PROPORTIONS:
        return target_counts[value.id], (value.target_ratio or 0.0) + 0.10

    equal_share = total_target / k
    desired = max(1, math.ceil(equal_share * balanced_floor_fraction))
    return desired, balanced_over_fraction / k


def coverage_gaps(
    distribution: TaskDistribution,
    state: GenerationState,
    total_target: int,
    *,
    balanced_floor_fraction: float = 0.70,
    balanced_over_fraction: float = 1.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return under- and overrepresented axis values using marginal coverage."""

    under: list[dict[str, Any]] = []
    over: list[dict[str, Any]] = []

    for axis in distribution.axes:
        counts = state.axis_counts.get(axis.name, {})
        observed_total = sum(counts.values())
        k = len(axis.values)

        target_counts = (
            _target_counts(axis, total_target)
            if axis.strategy == AxisStrategy.TARGET_PROPORTIONS
            else {}
        )

        for value in axis.values:
            actual = counts.get(value.id, 0)
            desired, allowed_share = _desired_and_allowed_share(
                axis, value, target_counts, k, total_target,
                balanced_floor_fraction, balanced_over_fraction,
            )

            if (gap := desired - actual) > 0:
                under.append(_axis_entry(axis, value, gap=gap))

            if observed_total > 0 and (share := actual / observed_total) > allowed_share:
                over.append(_axis_entry(axis, value, share=share))

    under.sort(key=lambda item: (-int(item["gap"]), str(item["axis"]), str(item["value_id"])))
    over.sort(key=lambda item: (-float(item["share"]), str(item["axis"]), str(item["value_id"])))

    return under, over


def _target(
    count: int,
    axis: str | None = None,
    value_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    constraints = (
        [{"axis": axis, "value_id": value_id, "description": description}]
        if axis is not None
        else []
    )
    return {"count": count, "constraints": constraints}


def build_generation_targets(
    distribution: TaskDistribution,
    state: GenerationState,
    *,
    batch_size: int,
    remaining_budget: int,
    total_target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a deterministic target plan from current marginal coverage gaps."""

    n = min(batch_size, remaining_budget)
    if n <= 0:
        return [], []

    under, over = coverage_gaps(distribution, state, total_target)

    if not under:
        return [_target(n)], over

    targets: list[dict[str, Any]] = []
    remaining = n
    used_axes: set[str] = set()

    per_axis_cap = max(1, math.ceil(n / max(1, len(distribution.axes))))

    for item in under:
        axis = str(item["axis"])
        if axis in used_axes or remaining <= 0:
            continue

        count = min(int(item["gap"]), per_axis_cap, remaining)
        targets.append(_target(count, axis, str(item["value_id"]), str(item["description"])))
        used_axes.add(axis)
        remaining -= count

    if remaining > 0:
        for item in under:
            if remaining <= 0:
                break

            axis = str(item["axis"])
            value_id = str(item["value_id"])

            already = sum(
                t["count"]
                for t in targets
                if t["constraints"]
                and t["constraints"][0]["axis"] == axis
                and t["constraints"][0]["value_id"] == value_id
            )

            extra_gap = max(0, int(item["gap"]) - already)
            if extra_gap <= 0:
                continue

            count = min(extra_gap, remaining)
            targets.append(_target(count, axis, value_id, str(item["description"])))
            remaining -= count

    if remaining > 0:
        targets.append(_target(remaining))

    return targets, over