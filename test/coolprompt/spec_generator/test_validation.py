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


def test_exact_pair_deduplication_normalizes_text_and_numbers() -> None:
    examples = [
        Example(input="  Price\u00a0value ", output="1.0"),
        Example(input="price value", output="1"),
        Example(input="Different", output="1"),
    ]

    assert Deduplicator.dedupe_exact_pairs_within_batch(examples) == [
        examples[0],
        examples[2],
    ]


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
        deduplicator=Deduplicator(enable_near_dup=False),
        max_topup_attempts=2,
    )

    result = pipeline.run(
        producer=producer,
        context=_classification_context(),
        target_n=2,
    )

    assert requested == [2, 1]
    assert [example.input for example in result] == ["first", "second"]
