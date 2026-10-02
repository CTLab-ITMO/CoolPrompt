from coolprompt.spec_generator import generate_problem_description
import pytest

from coolprompt.spec_generator.utils.retry import RetryConfig
from coolprompt.utils.enums import Task


class _SpecModel:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def invoke(self, request: str) -> dict[str, object]:
        self.requests.append(request)
        classification = "Classify sentiment" in request
        return {
            "description": (
                "Classify the input sentiment."
                if classification
                else "Summarize the supplied article."
            ),
        }


def test_generates_compact_description_from_prompt_and_examples() -> None:
    model = _SpecModel()

    description = generate_problem_description(
        model,
        "Summarize the article.",
        task=Task.GENERATION,
        examples=[("Long article", "Short summary")],
    )

    assert description == "Summarize the supplied article."
    assert len(model.requests) == 1
    assert "Long article" in model.requests[0]
    assert "input_format" not in model.requests[0]


def test_classification_labels_are_applied_when_supplied() -> None:
    model = _SpecModel()

    description = generate_problem_description(
        model,
        "Classify sentiment.",
        task=Task.CLASSIFICATION,
        labels=("positive", "negative"),
    )

    assert description == "Classify the input sentiment."
    assert "positive" in model.requests[0]


def test_invalid_description_response_is_retried() -> None:
    class _RetryModel:
        def __init__(self) -> None:
            self.outputs = iter(({"description": ""}, {"description": "Valid task."}))
            self.calls = 0

        def invoke(self, request: str) -> dict[str, str]:
            self.calls += 1
            return next(self.outputs)

    model = _RetryModel()
    description = generate_problem_description(
        model,
        "Perform the task.",
        task=Task.GENERATION,
        retry_config=RetryConfig(
            max_retries=1,
            min_wait_seconds=0,
            max_wait_seconds=0,
        ),
    )

    assert description == "Valid task."
    assert model.calls == 2


def test_rejects_labels_for_generation_task() -> None:
    with pytest.raises(ValueError, match="only valid for classification"):
        generate_problem_description(
            _SpecModel(),
            "Summarize text.",
            task=Task.GENERATION,
            labels=("summary",),
        )
