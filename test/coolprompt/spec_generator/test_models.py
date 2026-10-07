import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from coolprompt.spec_generator import (
    Example,
    GenerationContext,
    GenerationResult,
    TaskSpec,
    TaskSpecDraft,
)
from coolprompt.spec_generator.utils.model_utils import (
    StructuredResponseError,
    parse_structured,
)
from coolprompt.utils.enums import Task


class _FakeResponse(BaseModel):
    value: str


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


def test_parse_structured_falls_back_for_fake_list_llm() -> None:
    model = FakeListLLM(
        responses=['{"value": "fallback works"}'],
    )

    result = parse_structured(
        model=model,
        request="Return a structured response.",
        schema=_FakeResponse,
    )

    assert result == _FakeResponse(value="fallback works")


def test_parse_structured_falls_back_when_method_is_absent() -> None:
    class InvokeOnlyModel:
        def invoke(self, request: str) -> str:
            assert request == "Return a structured response."
            return '{"value": "plain invoke works"}'

    result = parse_structured(
        model=InvokeOnlyModel(),
        request="Return a structured response.",
        schema=_FakeResponse,
    )

    assert result == _FakeResponse(value="plain invoke works")


def test_parse_structured_uses_supported_structured_output() -> None:
    class Runnable:
        def invoke(self, request: str) -> dict[str, str]:
            assert request == "request"
            return {"value": "structured works"}

    class Model:
        def __init__(self) -> None:
            self.binding = None

        def with_structured_output(self, *, schema, method):
            self.binding = (schema, method)
            return Runnable()

    model = Model()
    result = parse_structured(model, "request", _FakeResponse)

    assert result == _FakeResponse(value="structured works")
    assert model.binding == (_FakeResponse, "json_schema")


def test_parse_structured_parses_ai_message_content() -> None:
    class Model:
        def invoke(self, request: str) -> AIMessage:
            return AIMessage(content='{"value": "message works"}')

    assert parse_structured(Model(), "request", _FakeResponse) == _FakeResponse(
        value="message works"
    )


def test_parse_structured_wraps_parse_and_validation_errors() -> None:
    class Model:
        def __init__(self, response: str) -> None:
            self.response = response

        def invoke(self, request: str) -> str:
            return self.response

    with pytest.raises(StructuredResponseError, match="failed validation"):
        parse_structured(Model("not-json"), "request", _FakeResponse)

    with pytest.raises(StructuredResponseError, match="failed validation"):
        parse_structured(Model('{"wrong": "field"}'), "request", _FakeResponse)
