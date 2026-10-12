from typing import cast

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.spec_generator import Example, SpecBuilder, TaskSpecDraft
from coolprompt.spec_generator.spec_builder import _apply_draft, _build_request
from coolprompt.spec_generator.utils.retry import RetryConfig
from coolprompt.utils.enums import Task


class _SequenceModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = iter(outputs)
        self.requests: list[str] = []

    def invoke(self, request: str) -> dict[str, object]:
        self.requests.append(request)
        return next(self._outputs)


def _as_model(model: _SequenceModel) -> BaseLanguageModel:
    """Cast the lightweight test double to the production model interface."""
    return cast(BaseLanguageModel, model)


def _spec_payload(
    *,
    task: str = "generation",
    labels: list[str] | None = None,
    description: str = "Summarize the input.",
) -> dict[str, object]:
    return {
        "task": task,
        "description": description,
        "input_format": "Text",
        "output_format": "Answer",
        "requirements": [],
        "labels": labels,
        "language": "English",
    }


def test_builder_includes_trusted_examples_and_skips_detection(monkeypatch) -> None:
    model = _SequenceModel([_spec_payload()])

    def unexpected_detection(self, prompt: str) -> None:
        raise AssertionError("dataset detection must not run")

    monkeypatch.setattr(SpecBuilder, "_detect_dataset", unexpected_detection)

    context = SpecBuilder(_as_model(model)).build(
        prompt="Summarize text.",
        examples=[("Long input", "Short output")],
        draft=TaskSpecDraft(task=Task.GENERATION),
        detect_dataset=False,
    )

    assert context.spec.description == "Summarize the input."
    assert context.seed_examples == (
        Example(input="Long input", output="Short output"),
    )
    assert "Long input" in model.requests[0]


def test_draft_overrides_model_fields_and_clears_stale_labels() -> None:
    model = _SequenceModel(
        [
            _spec_payload(
                task="classification",
                labels=["negative", "positive"],
            )
        ]
    )

    classification = (
        SpecBuilder(_as_model(model))
        .build(
            prompt="Classify text.",
            draft=TaskSpecDraft(
                task=Task.CLASSIFICATION,
                description="User description.",
                labels=("negative", "positive"),
            ),
        )
        .spec
    )

    generation = _apply_draft(
        classification,
        TaskSpecDraft(task=Task.GENERATION),
    )

    assert classification.description == "User description."
    assert generation.task == Task.GENERATION
    assert generation.labels is None


def test_validation_after_draft_application_is_retried() -> None:
    model = _SequenceModel(
        [
            _spec_payload(task="generation"),
            _spec_payload(
                task="classification",
                labels=["negative", "positive"],
                description="Classify sentiment.",
            ),
        ]
    )

    builder = SpecBuilder(
        _as_model(model),
        retry_config=RetryConfig(
            max_retries=1,
            min_wait_seconds=0,
            max_wait_seconds=0,
        ),
    )

    context = builder.build(
        prompt="Classify sentiment.",
        draft=TaskSpecDraft(task=Task.CLASSIFICATION),
    )

    assert context.spec.task == Task.CLASSIFICATION
    assert context.spec.labels == ("negative", "positive")
    assert len(model.requests) == 2


def test_spec_requests_define_when_empty_output_is_valid() -> None:
    for examples in ((), (Example(input="Question", output=""),)):
        request = _build_request(
            "Answer the question.",
            examples,
            None,
            None,
        )

        assert (
            "allow_empty_output: true only if an empty string " "is a valid task output"
        ) in request


def test_builder_preserves_inferred_empty_output_policy() -> None:
    model = _SequenceModel([_spec_payload() | {"allow_empty_output": True}])

    context = SpecBuilder(_as_model(model)).build(
        prompt="Return an empty string when no answer is supported.",
        detect_dataset=False,
    )

    assert context.spec.allow_empty_output is True


def test_detected_generation_dataset_supplies_reference_examples(monkeypatch) -> None:
    model = _SequenceModel([_spec_payload(description="Write a concept sentence.")])
    monkeypatch.setattr(
        SpecBuilder,
        "_detect_dataset",
        lambda self, prompt: "common_gen",
    )

    context = SpecBuilder(_as_model(model)).build(
        prompt="Use all concepts in a natural sentence.",
        detect_dataset=True,
    )

    assert context.dataset_name == "common_gen"
    assert context.seed_examples
    assert "dog" in model.requests[0]


def test_detected_classification_dataset_is_discarded_when_labels_do_not_match(
    monkeypatch,
) -> None:
    model = _SequenceModel(
        [
            _spec_payload(
                task="classification",
                labels=["negative", "positive"],
                description="Classify sentiment.",
            )
        ]
    )
    monkeypatch.setattr(
        SpecBuilder,
        "_detect_dataset",
        lambda self, prompt: "tweeteval",
    )

    context = SpecBuilder(_as_model(model)).build(
        prompt="Classify sentiment.",
        detect_dataset=True,
    )

    assert context.dataset_name is None
    assert context.seed_examples == ()


def test_explicit_examples_take_priority_over_detected_dataset_examples(
    monkeypatch,
) -> None:
    model = _SequenceModel([_spec_payload()])
    monkeypatch.setattr(
        SpecBuilder,
        "_detect_dataset",
        lambda self, prompt: "gsm8k",
    )

    context = SpecBuilder(_as_model(model)).build(
        prompt="Solve the problem.",
        examples=[("custom input", "custom output")],
        detect_dataset=True,
    )

    assert context.seed_examples == (
        Example(input="custom input", output="custom output"),
    )
