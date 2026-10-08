"""Utilities for structured model invocation, parsing, and fallback."""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, ValidationError

from coolprompt.utils.parsing import extract_json

_T = TypeVar("_T", bound=BaseModel)


class StructuredResponseError(ValueError):
    """A model response could not be decoded or validated."""


def invoke_structured(
    model: BaseLanguageModel,
    request: str,
    schema: object,
    *,
    method: str = "json_schema",
    error_cls: type[ValueError] = StructuredResponseError,
    label: str = "Structured",
) -> Any:
    """Invoke the model with structured output and fall back when unsupported."""

    bind = getattr(model, "with_structured_output", None)

    if bind is None:
        runnable = model
    else:
        try:
            runnable = bind(schema=schema, method=method)
        except NotImplementedError:
            runnable = model

    try:
        output = runnable.invoke(request)
    except ValidationError as exc:
        raise error_cls(f"{label} response failed validation.") from exc

    if isinstance(output, AIMessage):
        output = output.content

    if isinstance(output, str):
        try:
            return extract_json(output)
        except (TypeError, ValueError) as exc:
            raise error_cls(f"{label} response could not be parsed.") from exc

    return output


def parse_structured(
    model: BaseLanguageModel,
    request: str,
    schema: type[_T],
    *,
    method: str = "json_schema",
    error_cls: type[ValueError] = StructuredResponseError,
    label: str = "Structured",
) -> _T:
    """Invoke the model and validate the response with a Pydantic schema."""

    output = invoke_structured(
        model,
        request,
        schema,
        method=method,
        error_cls=error_cls,
        label=label,
    )

    if isinstance(output, schema):
        return output

    try:
        return schema.model_validate(output)
    except ValidationError as exc:
        raise error_cls(f"{label} response failed validation.") from exc
