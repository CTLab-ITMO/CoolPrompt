"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.logging_config import logger
from coolprompt.utils.default import (
    DEFAULT_HF_ENDPOINT_MODEL_NAME,
    DEFAULT_HF_ENDPOINT_PROVIDER,
    DEFAULT_HF_MODEL_PARAMETERS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PARAMETERS,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OUTLINES_MODEL_PARAMETERS,
)


class DefaultLLM:
    """Default LangChain-compatible LLM using transformers."""

    @staticmethod
    def init(
        langchain_provider: str = "vllm",
        **kwargs,
    ) -> BaseLanguageModel:
        """Initializing the default LLM via given provider

        Args:
            langchain_provider (str): provider used for model's initialization.
            Available providers: `vllm` (default), `hf_endpoint`,
            `hf_pipeline`, `llamacpp`, `ollama`, `gpt4all`,
            `outlines`."""

        provider = langchain_provider
        logger.info(f"Initializing with provider: {provider}")
        provider_map = {
            "vllm": DefaultLLM.init_vllm,
            "hf_endpoint": DefaultLLM.init_hf_endpoint,
            "hf_pipeline": DefaultLLM.init_hf_pipeline,
            "llamacpp": DefaultLLM.init_llamacpp,
            "ollama": DefaultLLM.init_ollama,
            "gpt4all": DefaultLLM.init_gpt4all,
            "outlines": DefaultLLM.init_outlines,
        }

        if provider not in provider_map:
            error_msg = (
                f"Invalid provider '{provider}'. Allowed providers: "
                f"{list(provider_map.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Itinializing with given kwargs: {kwargs}")
        return provider_map[provider](**kwargs)

    @staticmethod
    def init_vllm(
        langchain_config: dict[str, any] | None = None,
        vllm_engine_config: dict[str, any] | None = None,
    ) -> BaseLanguageModel:
        """Initialize the transformers-powered LangChain LLM.

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
        logger.info(f"Initializing default model: {DEFAULT_MODEL_NAME}")
        generation_and_model_config = DEFAULT_MODEL_PARAMETERS.copy()
        if langchain_config is not None:
            generation_and_model_config.update(langchain_config)

        llm = HuggingFacePipeline.from_model_id(
            model_id=DEFAULT_MODEL_NAME,
            task="text-generation",
            pipeline_kwargs=generation_and_model_config
        )
        return ChatHuggingFace(llm=llm)
