from coolprompt.spec_generator import Example, GenerationContext, TaskSpec
from coolprompt.spec_generator.validation.format import Deduplicator, ExampleValidator
from coolprompt.spec_generator.validation.pipeline import ValidationPipeline
from coolprompt.utils.enums import Task


def _classification_context() -> GenerationContext:
    return GenerationContext(
        spec=TaskSpec(
            task=Task.CLASSIFICATION,
            description="Classify sentiment.",
            input_format="Review text.",
            output_format="One label.",
            labels=("Negative", "Positive"),
        )
    )


def test_validator_normalizes_labels_and_rejects_unknown_values() -> None:
    valid, invalid = ExampleValidator().validate(
        [
            {"input": "Great product", "output": "positive"},
            {"input": "Unclear product", "output": "neutral"},
            {"input": "", "output": "negative"},
        ],
        _classification_context().spec,
    )

    assert valid == [Example(input="Great product", output="Positive")]
    assert len(invalid) == 2


def test_pipeline_keeps_valid_examples_and_only_tops_up_missing_count() -> None:
    requested: list[int] = []
    responses = iter(
        [
            [
                {"input": "first", "output": "positive"},
                {"input": "bad", "output": "unknown"},
            ],
            [
                {"input": "second", "output": "negative"},
            ],
        ]
    )

    def producer(remaining: int) -> list[dict[str, str]]:
        requested.append(remaining)
        return next(responses)

    pipeline = ValidationPipeline(
        validator=ExampleValidator(),
        deduplicator=Deduplicator(),
        max_topup_attempts=2,
    )

    result = pipeline.run(
        producer=producer,
        context=_classification_context(),
        target_n=2,
    )

    assert requested == [2, 1]
    assert [example.input for example in result] == ["first", "second"]


def test_validator_applies_empty_output_policy() -> None:
    spec = _classification_context().spec.model_copy(
        update={"task": Task.GENERATION, "labels": None}
    )

    valid, invalid = ExampleValidator().validate(
        [{"input": "Question", "output": ""}],
        spec,
    )
    assert valid == []
    assert len(invalid) == 1

    valid, invalid = ExampleValidator().validate(
        [{"input": "Question", "output": " \n\t"}],
        spec.model_copy(update={"allow_empty_output": True}),
    )
    assert valid == [Example(input="Question", output="")]
    assert invalid == []


def test_deduplicator_rejects_exact_and_near_duplicate_inputs() -> None:
    deduplicator = Deduplicator()

    assert deduplicator.accept(
        Example(
            input="The quick brown fox jumps over the lazy dog near the river bank.",
            output="a",
        )
    )
    assert not deduplicator.accept(
        Example(
            input="  THE quick brown fox jumps over the lazy dog near the river bank. ",
            output="b",
        )
    )
    assert not deduplicator.accept(
        Example(
            input="The quick brown fox jumps over a lazy dog near the river bank.",
            output="c",
        )
    )
    assert deduplicator.accept(
        Example(input="Photosynthesis converts light into chemical energy.", output="d")
    )


def test_pipeline_stops_after_topup_budget_when_producer_is_empty() -> None:
    calls = 0

    def producer(remaining: int) -> list[object]:
        nonlocal calls
        calls += 1
        return []

    pipeline = ValidationPipeline(
        validator=ExampleValidator(),
        deduplicator=Deduplicator(),
        max_topup_attempts=3,
    )

    assert pipeline.run(producer, _classification_context(), 1) == []
    assert calls == 3


def test_pipeline_invokes_acceptance_callback_only_for_accepted_examples() -> None:
    accepted: list[str] = []
    pipeline = ValidationPipeline(
        validator=ExampleValidator(),
        deduplicator=Deduplicator(),
        max_topup_attempts=1,
    )

    result = pipeline.run(
        lambda _: [
            {"input": "keep", "output": "positive"},
            {"input": "reject", "output": "negative"},
        ],
        _classification_context(),
        1,
        accept_candidate=lambda example: example.input == "keep",
        on_accept=lambda example: accepted.append(example.input),
    )

    assert [example.input for example in result] == ["keep"]
    assert accepted == ["keep"]
