import pytest
from pydantic import ValidationError

from coolprompt.spec_generator import (
    Example,
    GenerationContext,
    GenerationResult,
    TaskSpec,
    TaskSpecDraft,
)
from coolprompt.utils.enums import Task


def _classification_spec() -> TaskSpec:
    return TaskSpec(
        task=Task.CLASSIFICATION,
        description="Classify sentiment.",
        input_format="One review.",
        output_format="One label.",
        labels=("negative", "positive"),
    )


def test_classification_requires_labels() -> None:
    with pytest.raises(ValidationError, match="require at least one label"):
        TaskSpec(
            task=Task.CLASSIFICATION,
            description="Classify sentiment.",
            input_format="One review.",
            output_format="One label.",
        )


def test_generation_rejects_labels() -> None:
    with pytest.raises(ValidationError, match="only valid for classification"):
        TaskSpec(
            task=Task.GENERATION,
            description="Summarize text.",
            input_format="Article text.",
            output_format="One sentence.",
            labels=("summary",),
        )


def test_spec_normalizes_case_insensitive_collections() -> None:
    spec = TaskSpec(
        task=Task.CLASSIFICATION,
        description="Classify sentiment.",
        input_format="One review.",
        output_format="One label.",
        requirements=("Be concise", " be concise ", "Return one label"),
        labels=("Positive", " positive ", "Negative"),
    )

    assert spec.requirements == ("Be concise", "Return one label")
    assert spec.labels == ("Positive", "Negative")


def test_draft_tracks_only_explicit_overrides() -> None:
    draft = TaskSpecDraft(description="User-provided description.")

    assert draft.is_empty is False
    assert draft.overrides() == {"description": "User-provided description."}
    assert TaskSpecDraft().is_empty is True


def test_generation_result_projects_inputs_and_outputs() -> None:
    result = GenerationResult(
        examples=(
            Example(input="first", output="negative"),
            Example(input="second", output="positive"),
        ),
        context=GenerationContext(spec=_classification_spec()),
    )

    assert result.dataset == ["first", "second"]
    assert result.target == ["negative", "positive"]
