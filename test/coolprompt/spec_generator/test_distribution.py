import random

import pytest
from pydantic import ValidationError

from coolprompt.spec_generator.distribution import (
    AxisStrategy,
    AxisValue,
    GenerationState,
    TaskAxis,
    TaskDistribution,
    _allocate_proportional,
    _allocate_quotas,
    axis_quotas,
    build_generation_targets,
    coverage_gaps,
    trim_indices,
    validate_axis_tags,
)
from coolprompt.spec_generator.models import TaskSpec
from coolprompt.utils.enums import Task


def _distribution() -> TaskDistribution:
    return TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Problem difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy case."),
                    AxisValue(id="hard", description="Hard case."),
                ),
            ),
            TaskAxis(
                name="input_size",
                description="Number of concepts.",
                strategy=AxisStrategy.TARGET_PROPORTIONS,
                values=(
                    AxisValue(
                        id="size:2",
                        description="Two concepts.",
                        target_ratio=0.5,
                    ),
                    AxisValue(
                        id="size:3",
                        description="Three concepts.",
                        target_ratio=0.5,
                    ),
                ),
            ),
            TaskAxis(
                name="label",
                description="Classification label.",
                strategy=AxisStrategy.TARGET_PROPORTIONS,
                values=(
                    AxisValue(
                        id="label:0",
                        description="Negative.",
                        target_ratio=0.5,
                    ),
                    AxisValue(
                        id="label:1",
                        description="Positive.",
                        target_ratio=0.5,
                    ),
                ),
            ),
        )
    )


def test_axis_tags_validate_model_values_and_override_label_axis() -> None:
    spec = TaskSpec(
        task=Task.CLASSIFICATION,
        description="Classify sentiment.",
        input_format="A list of concepts.",
        output_format="One label.",
        labels=("negative", "positive"),
    )

    tags = validate_axis_tags(
        _distribution(),
        {
            "Difficulty": "hard",
            "input-size": "size:99",
            "label": "label:0",
            "unknown": "value",
        },
        output=" POSITIVE ",
        spec=spec,
    )

    assert tags == {
        "difficulty": "hard",
        "label": "label:1",
    }


def test_coverage_gaps_reports_under_and_overrepresented_values() -> None:
    distribution = TaskDistribution(axes=(_distribution().axis("difficulty"),))
    state = GenerationState(axis_counts={"difficulty": {"easy": 4, "hard": 0}})

    under, over = coverage_gaps(distribution, state, total_target=4)

    assert any(item["value_id"] == "hard" for item in under)
    assert any(item["value_id"] == "easy" for item in over)


def test_balanced_quotas_require_exact_near_equal_counts() -> None:
    axis = _distribution().axis("difficulty")
    distribution = TaskDistribution(axes=(axis,))

    assert axis_quotas(distribution, 100)["difficulty"] == {
        "easy": (50, 0.5),
        "hard": (50, 0.5),
    }

    state = GenerationState(axis_counts={"difficulty": {"easy": 70, "hard": 30}})
    under, over = coverage_gaps(distribution, state, 100)

    assert [(item["value_id"], item["gap"]) for item in under] == [("hard", 20)]
    assert [item["value_id"] for item in over] == ["easy"]


def test_balanced_quotas_assign_remainders_deterministically() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="kind",
                description="Example kind.",
                values=tuple(
                    AxisValue(id=value, description=value)
                    for value in ("first", "second", "third")
                ),
            ),
        )
    )

    assert {
        value: quota[0] for value, quota in axis_quotas(distribution, 5)["kind"].items()
    } == {"first": 2, "second": 2, "third": 1}

    state = GenerationState(axis_counts={"kind": {"first": 2, "second": 1}})
    _, over = coverage_gaps(distribution, state, 5)
    assert over == []


def test_quotas_and_slot_allocation_respect_target_counts() -> None:
    distribution = TaskDistribution(axes=(_distribution().axis("input_size"),))
    quotas = axis_quotas(distribution, 5)["input_size"]

    assert [quotas[value][0] for value in ("size:2", "size:3")] == [3, 2]
    assert _allocate_quotas([5, 3, 0], 4) == [3, 1, 0]
    assert _allocate_quotas([1, 2], 5) == [1, 2]


def test_post_target_allocation_oversamples_gaps_proportionally() -> None:
    assert _allocate_proportional([1, 3], 8) == [2, 6]
    assert _allocate_proportional([1, 1, 1], 2) == [1, 1, 0]
    assert _allocate_proportional([0, 0], 5) == [0, 0]


def test_post_target_generation_uses_only_targeted_slots() -> None:
    distribution = TaskDistribution(axes=(_distribution().axis("difficulty"),))
    state = GenerationState(axis_counts={"difficulty": {"easy": 10}})

    targets, _ = build_generation_targets(
        distribution,
        state,
        batch_size=10,
        remaining_budget=10,
        total_target=10,
        post_target=True,
        mix_axes=False,
    )

    assert targets == [
        {
            "count": 10,
            "constraints": [
                {
                    "axis": "difficulty",
                    "value_id": "hard",
                    "description": "Hard case.",
                }
            ],
        }
    ]


def test_unrecord_failure_preserves_state() -> None:
    state = GenerationState()
    state.record({"difficulty": "easy", "input_size": "size:2"})

    with pytest.raises(ValueError, match="missing"):
        state.unrecord({"difficulty": "easy", "missing": "value"})

    assert state.axis_counts == {
        "difficulty": {"easy": 1},
        "input_size": {"size:2": 1},
    }
    state.unrecord({"difficulty": "easy", "input_size": "size:2"})
    assert state.axis_counts == {}


def test_targets_do_not_require_unknown_cross_axis_combinations() -> None:
    distribution = TaskDistribution(axes=_distribution().axes[:2])
    state = GenerationState(
        axis_counts={
            "difficulty": {"easy": 3},
            "input_size": {"size:2": 3},
        }
    )

    targets, _ = build_generation_targets(
        distribution,
        state,
        batch_size=2,
        remaining_budget=2,
        total_target=4,
        rng=random.Random(0),
    )

    assert sum(target["count"] for target in targets) == 2
    assert all(len(target["constraints"]) <= 1 for target in targets)
    assert {
        (item["axis"], item["value_id"])
        for target in targets
        for item in target["constraints"]
    } == {("difficulty", "hard"), ("input_size", "size:3")}


def test_trim_preserves_feasible_coverage_across_axes() -> None:
    distribution = TaskDistribution(
        axes=tuple(
            TaskAxis(
                name=axis,
                description=axis,
                values=(
                    AxisValue(id="0", description="First."),
                    AxisValue(id="1", description="Second."),
                ),
            )
            for axis in ("a", "b")
        )
    )
    tags = [
        {"a": "0", "b": "0"},
        {"a": "0", "b": "1"},
        {"a": "0", "b": "1"},
        {"a": "1", "b": "0"},
        {"a": "1", "b": "0"},
    ]
    state = GenerationState()
    for item in tags:
        state.record(item)

    dropped = trim_indices(tags, 2, axis_quotas(distribution, 3), state.axis_counts, 3)
    for index in dropped:
        state.unrecord(tags[index])

    assert len(dropped) == 2
    assert not coverage_gaps(distribution, state, 3)[0]
    assert state.axis_counts == {
        "a": {"0": 2, "1": 1},
        "b": {"0": 2, "1": 1},
    }


def test_axis_models_reject_invalid_values_and_ratios() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        TaskAxis(
            name="kind",
            description="Kind.",
            values=(AxisValue(id="one", description="One."),),
        )

    with pytest.raises(ValidationError, match="unique"):
        TaskAxis(
            name="kind",
            description="Kind.",
            values=(
                AxisValue(id="same", description="One."),
                AxisValue(id="SAME", description="Two."),
            ),
        )

    with pytest.raises(ValidationError, match="must not define"):
        TaskAxis(
            name="kind",
            description="Kind.",
            values=(
                AxisValue(id="a", description="A.", target_ratio=0.5),
                AxisValue(id="b", description="B."),
            ),
        )

    with pytest.raises(ValidationError, match="sum approximately"):
        TaskAxis(
            name="kind",
            description="Kind.",
            strategy=AxisStrategy.TARGET_PROPORTIONS,
            values=(
                AxisValue(id="a", description="A.", target_ratio=0.2),
                AxisValue(id="b", description="B.", target_ratio=0.2),
            ),
        )


def test_distribution_rejects_equivalent_axis_names() -> None:
    axes = tuple(
        TaskAxis(
            name=name,
            description="Size.",
            values=(
                AxisValue(id="small", description="Small."),
                AxisValue(id="large", description="Large."),
            ),
        )
        for name in ("input_size", "Input-Size")
    )

    with pytest.raises(ValidationError, match="unique"):
        TaskDistribution(axes=axes)


def test_target_proportion_quotas_use_largest_remainders() -> None:
    axis = TaskAxis(
        name="kind",
        description="Kind.",
        strategy=AxisStrategy.TARGET_PROPORTIONS,
        values=(
            AxisValue(id="a", description="A.", target_ratio=0.5),
            AxisValue(id="b", description="B.", target_ratio=0.3),
            AxisValue(id="c", description="C.", target_ratio=0.2),
        ),
    )

    assert {
        value: quota[0]
        for value, quota in axis_quotas(
            TaskDistribution(axes=(axis,)),
            7,
        )["kind"].items()
    } == {"a": 4, "b": 2, "c": 1}


def test_trim_rejects_inconsistent_state_and_target_size() -> None:
    distribution = TaskDistribution(axes=(_distribution().axis("difficulty"),))
    tags = [{"difficulty": "easy"}, {"difficulty": "hard"}]
    quotas = axis_quotas(distribution, 1)

    with pytest.raises(ValueError, match="total_target"):
        trim_indices(tags, 1, quotas, {"difficulty": {"easy": 1, "hard": 1}}, 2)

    with pytest.raises(ValueError, match="differs"):
        trim_indices(tags, 1, quotas, {"difficulty": {"easy": 2}}, 1)
