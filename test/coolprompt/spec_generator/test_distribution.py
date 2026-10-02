from coolprompt.spec_generator.distribution import (
    AxisStrategy,
    AxisValue,
    GenerationState,
    TaskAxis,
    TaskDistribution,
    coverage_gaps,
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
