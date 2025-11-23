"""LangChain-compatible LLM interface."""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.logging_config import logger
from coolprompt.utils.default import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PARAMETERS,
)


class DefaultLLM:
    """Default LangChain-compatible LLM using transformers."""

    @staticmethod
    def init(
            langchain_config: dict[str, any] | None = None,
    ) -> BaseLanguageModel:
        """Initialize the transformers-powered LangChain LLM.

        Args:
            langchain_config (dict[str, Any], optional):
                Optional dictionary of LangChain VLLM parameters
                (temperature, top_p, etc).
                Overrides DEFAULT_MODEL_PARAMETERS.
        Returns:
            BaseLanguageModel:
                Initialized LangChain-compatible language model instance.
        """
        logger.info(f"Initializing default model: {DEFAULT_MODEL_NAME}")
        generation_and_model_config = DEFAULT_MODEL_PARAMETERS.copy()
        if langchain_config is not None:
            generation_and_model_config.update(langchain_config)

        llm = HuggingFacePipeline.from_model_id(
            model_id=DEFAULT_MODEL_NAME,
            task="text-generation",
            pipeline_kwargs=generation_and_model_config,
            model_kwargs={'dtype': 'float16'}
        )
        return ChatHuggingFace(llm=llm)
