"""Utilities for resolving LangChain chat models."""

from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel


def resolve_chat_model(model: BaseLanguageModel) -> BaseLanguageModel | None:
    """Return a model that supports structured output without unwrapping it."""

    if hasattr(model, "with_structured_output"):
        return model

    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "with_structured_output"):
        return wrapped

    return None
