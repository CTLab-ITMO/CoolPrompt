"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

from langchain_community.llms import VLLM
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.logging_config import logger
from coolprompt.utils.default import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PARAMETERS,
)


class DefaultLLM:
    """Default LangChain-compatible LLM using vLLM engine."""

    @staticmethod
    def init(
        langchain_config: dict[str, any] | None = None,
        vllm_engine_config: dict[str, any] | None = None,
    ) -> BaseLanguageModel:
        """Initialize the vLLM-powered LangChain LLM.

        Args:
            langchain_config (dict[str, Any], optional):
                Optional dictionary of LangChain VLLM parameters
                (temperature, top_p, etc).
                Overrides DEFAULT_MODEL_PARAMETERS.
            vllm_engine_config (dict[str, Any], optional):
                Optional dictionary of low-level vllm.LLM parameters
                (gpu_memory_utilization, max_model_len, etc).
                Passed directly to vllm.LLM via vllm_kwargs.
        Returns:
            BaseLanguageModel:
                Initialized LangChain-compatible language model instance.
        """
        logger.info("Initializing default model")
        logger.debug(
            "Updating default model params with "
            f"langchain config: {langchain_config} "
            f"and vllm_engine_config: {vllm_engine_config}"
        )
        generation_and_model_config = DEFAULT_MODEL_PARAMETERS.copy()
        if langchain_config is not None:
            generation_and_model_config.update(langchain_config)

        return VLLM(
            model=DEFAULT_MODEL_NAME,
            trust_remote_code=True,
            dtype="float16",
            vllm_kwargs=vllm_engine_config,
            **generation_and_model_config,
        )
