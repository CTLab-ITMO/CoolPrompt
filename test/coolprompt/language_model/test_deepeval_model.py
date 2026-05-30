"""Unit tests for :mod:`coolprompt.language_model.deepeval_model`.

These tests cover the structured-output flag added to
:class:`coolprompt.language_model.deepeval_model.DeepEvalLangChainModel`:

* When ``use_structured_output=False`` (default), the wrapper keeps its
  legacy contract: it calls ``model.invoke`` / ``model.ainvoke`` and
  coerces the result into a ``str``.
* When ``use_structured_output=True`` and DeepEval supplies a pydantic
  ``schema`` (per the ``DeepEvalBaseLLM`` contract), the wrapper routes
  the call through ``model.with_structured_output(schema, method=
  "json_schema").invoke(prompt)`` and returns the populated pydantic
  instance verbatim.
* When ``use_structured_output=True`` and no schema is supplied, the
  wrapper falls back to ``DeepEvalJudgeResponse`` and returns only its
  ``response`` field as a ``str`` so DeepEval code expecting a string
  keeps working.
* :class:`coolprompt.evaluator.metrics.GEvalMetric` forwards the flag
  into :class:`DeepEvalLangChainModel`.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from coolprompt.language_model.deepeval_model import DeepEvalLangChainModel
from coolprompt.utils.structured_schemas.language_model import (
    DeepEvalJudgeResponse,
)


class _DummySchema(BaseModel):
    score: int


def _make_chat_model(invoke_return=None, ainvoke_return=None):
    """Build a MagicMock LangChain chat model with the methods we use."""
    model = MagicMock()
    model.invoke = MagicMock(return_value=invoke_return)

    async def _ainvoke(prompt):
        return ainvoke_return

    model.ainvoke = MagicMock(side_effect=_ainvoke)
    return model


# ---------------------------------------------------------------------------
# Legacy behaviour: use_structured_output=False
# ---------------------------------------------------------------------------


def test_generate_unstructured_returns_str_from_ai_message():
    chat = _make_chat_model(invoke_return=AIMessage(content="hello"))
    wrapper = DeepEvalLangChainModel(chat)

    result = wrapper.generate("prompt")

    assert result == "hello"
    chat.invoke.assert_called_once_with("prompt")
    chat.with_structured_output.assert_not_called()


def test_generate_unstructured_returns_str_from_plain_value():
    chat = _make_chat_model(invoke_return=42)
    wrapper = DeepEvalLangChainModel(chat)

    assert wrapper.generate("prompt") == "42"


def test_generate_unstructured_ignores_schema_argument():
    """When the flag is off, DeepEval's schema must NOT be applied."""
    chat = _make_chat_model(invoke_return=AIMessage(content="raw"))
    wrapper = DeepEvalLangChainModel(chat, use_structured_output=False)

    assert wrapper.generate("p", schema=_DummySchema) == "raw"
    chat.with_structured_output.assert_not_called()


def test_a_generate_unstructured_returns_str_from_ai_message():
    chat = _make_chat_model(ainvoke_return=AIMessage(content="hi"))
    wrapper = DeepEvalLangChainModel(chat)

    result = asyncio.get_event_loop().run_until_complete(
        wrapper.a_generate("prompt")
    )

    assert result == "hi"
    chat.with_structured_output.assert_not_called()


# ---------------------------------------------------------------------------
# Structured behaviour: use_structured_output=True with explicit schema
# ---------------------------------------------------------------------------


def test_generate_structured_with_schema_returns_pydantic_instance():
    chat = MagicMock()
    runner = MagicMock()
    parsed = _DummySchema(score=7)
    runner.invoke = MagicMock(return_value=parsed)
    chat.with_structured_output = MagicMock(return_value=runner)

    wrapper = DeepEvalLangChainModel(chat, use_structured_output=True)
    result = wrapper.generate("prompt", schema=_DummySchema)

    chat.with_structured_output.assert_called_once_with(
        _DummySchema, method="json_schema"
    )
    runner.invoke.assert_called_once_with("prompt")
    assert result is parsed
    chat.invoke.assert_not_called()


def test_a_generate_structured_with_schema_returns_pydantic_instance():
    chat = MagicMock()
    runner = MagicMock()
    parsed = _DummySchema(score=3)

    async def _ainvoke(prompt):
        return parsed

    runner.ainvoke = MagicMock(side_effect=_ainvoke)
    chat.with_structured_output = MagicMock(return_value=runner)

    wrapper = DeepEvalLangChainModel(chat, use_structured_output=True)
    result = asyncio.get_event_loop().run_until_complete(
        wrapper.a_generate("prompt", schema=_DummySchema)
    )

    chat.with_structured_output.assert_called_once_with(
        _DummySchema, method="json_schema"
    )
    assert result is parsed


# ---------------------------------------------------------------------------
# Structured behaviour: use_structured_output=True with no schema (fallback)
# ---------------------------------------------------------------------------


def test_generate_structured_no_schema_falls_back_and_unwraps_response():
    chat = MagicMock()
    runner = MagicMock()
    runner.invoke = MagicMock(
        return_value=DeepEvalJudgeResponse(response="plain")
    )
    chat.with_structured_output = MagicMock(return_value=runner)

    wrapper = DeepEvalLangChainModel(chat, use_structured_output=True)
    result = wrapper.generate("prompt")

    chat.with_structured_output.assert_called_once_with(
        DeepEvalJudgeResponse, method="json_schema"
    )
    assert result == "plain"


def test_a_generate_structured_no_schema_falls_back_and_unwraps_response():
    chat = MagicMock()
    runner = MagicMock()

    async def _ainvoke(prompt):
        return DeepEvalJudgeResponse(response="async-plain")

    runner.ainvoke = MagicMock(side_effect=_ainvoke)
    chat.with_structured_output = MagicMock(return_value=runner)

    wrapper = DeepEvalLangChainModel(chat, use_structured_output=True)
    result = asyncio.get_event_loop().run_until_complete(
        wrapper.a_generate("prompt")
    )

    assert result == "async-plain"


# ---------------------------------------------------------------------------
# GEvalMetric forwards the flag into the wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [True, False])
def test_geval_metric_forwards_flag_to_deepeval_wrapper(flag):
    """``GEvalMetric`` must propagate ``use_structured_output`` to the
    underlying :class:`DeepEvalLangChainModel`."""
    # Importing here keeps the fast tests independent of the heavy deepeval
    # GEval class until this specific test is collected.
    from coolprompt.evaluator import metrics as metrics_mod

    base_model = MagicMock()

    captured = {}

    class _Recorder(DeepEvalLangChainModel):
        def __init__(self, model, use_structured_output=False):
            super().__init__(model, use_structured_output=use_structured_output)
            captured["use_structured_output"] = use_structured_output

    with patch.object(
        metrics_mod, "DeepEvalLangChainModel", _Recorder
    ), patch.object(metrics_mod, "GEval", MagicMock()):
        metrics_mod.GEvalMetric(
            model=base_model,
            criteria="be helpful",
            use_structured_output=flag,
        )

    assert captured["use_structured_output"] is flag
