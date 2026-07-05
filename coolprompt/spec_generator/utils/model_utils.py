from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel


def resolve_chat_model(model: BaseLanguageModel) -> BaseChatModel | None:
    if isinstance(model, BaseChatModel):
        return model

    wrapped_model = getattr(model, "model", None)

    if isinstance(wrapped_model, BaseChatModel):
        return wrapped_model

    return None
