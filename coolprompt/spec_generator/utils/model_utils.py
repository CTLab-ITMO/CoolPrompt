"""Utilities for resolving LangChain chat models."""

from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel


def resolve_chat_model(model: BaseLanguageModel) -> BaseChatModel | None:
    """Return a chat model directly or through a common wrapper attribute."""

    if isinstance(model, BaseChatModel):
        return model

    wrapped = getattr(model, "model", None)
    return wrapped if isinstance(wrapped, BaseChatModel) else None
