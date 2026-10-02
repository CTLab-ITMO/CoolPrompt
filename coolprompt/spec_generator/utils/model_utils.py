"""Utilities for resolving LangChain chat models."""

from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel


def resolve_chat_model(model: BaseLanguageModel) -> BaseLanguageModel | None:
    """Return the original model if it supports structured output."""

    if hasattr(model, "with_structured_output"):
        return model

    return None
