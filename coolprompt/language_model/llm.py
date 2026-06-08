"""LangChain-compatible LLM interface."""

from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

from coolprompt.utils.default import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PARAMETERS,
)
from coolprompt.utils.logging_config import logger


class DefaultLLM:
    """Default LangChain-compatible LLM using the OpenAI API."""

    @staticmethod
    def init(
        langchain_config: dict[str, Any] | None = None,
    ) -> BaseLanguageModel:
        """Initialize the OpenAI-powered LangChain LLM.

        Args:
            langchain_config (dict[str, Any], optional):
                Optional dictionary of ChatOpenAI parameters
                (temperature, max_tokens, etc).
                Overrides DEFAULT_MODEL_PARAMETERS.
        Returns:
            BaseLanguageModel:
                Initialized LangChain-compatible language model instance
                based on OpenAI API.
        """
        logger.info(f"Initializing default model: {DEFAULT_MODEL_NAME}")
        model_config = DEFAULT_MODEL_PARAMETERS.copy()
        if langchain_config is not None:
            model_config.update(langchain_config)

        return ChatOpenAI(
            model=DEFAULT_MODEL_NAME,
            **model_config,
        )
