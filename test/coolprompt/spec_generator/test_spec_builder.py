from coolprompt.spec_generator import Example, SpecBuilder, TaskSpecDraft
from coolprompt.spec_generator.spec_builder import _apply_draft
from coolprompt.spec_generator.utils.retry import RetryConfig
from coolprompt.utils.enums import Task


class _SequenceModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = iter(outputs)
        self.requests: list[str] = []

    def invoke(self, request: str) -> dict[str, object]:
        self.requests.append(request)
        return next(self._outputs)


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

    context = SpecBuilder(model).build(
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
    classification = (
        SpecBuilder(
            _SequenceModel(
                [
                    _spec_payload(
                        task="classification",
                        labels=["negative", "positive"],
                    )
                ]
            )
        )
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
        model,
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
