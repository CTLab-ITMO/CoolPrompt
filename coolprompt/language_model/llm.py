"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

from pathlib import Path
from langchain_community.llms import VLLM, LlamaCpp, GPT4All, Outlines
from langchain_ollama.llms import OllamaLLM
import torch
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
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
    """Default LangChain-compatible LLM using vLLM engine."""

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

    @staticmethod
    def init_hf_endpoint(
        model_name: str = DEFAULT_HF_ENDPOINT_MODEL_NAME, **kwargs
    ) -> BaseLanguageModel:
        logger.debug("Initializing HF endpoint provider")
        config = {**DEFAULT_HF_MODEL_PARAMETERS, **kwargs}
        return HuggingFaceEndpoint(
            repo_id=model_name,
            provider=DEFAULT_HF_ENDPOINT_PROVIDER,
            **config,
        )

    @staticmethod
    def init_hf_pipeline(
        model_name: str = DEFAULT_MODEL_NAME, **kwargs
    ) -> BaseLanguageModel:
        logger.debug("Initializing HF Pipeline provider")
        from transformers import pipeline

        config = {**DEFAULT_HF_MODEL_PARAMETERS, **kwargs}
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **config,
        )
        return HuggingFacePipeline(pipeline=pipe)

    @staticmethod
    def init_llamacpp(model_path: str | Path, **kwargs) -> BaseLanguageModel:
        logger.debug("Initializing Llama.cpp provider")
        if not Path(model_path).exists():
            error_msg = (
                "Model path for Llama.cpp initialization does not exist"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        config = {**DEFAULT_MODEL_PARAMETERS, **kwargs}
        return LlamaCpp(model_path=model_path, **config)

    @staticmethod
    def init_ollama(
        model_name: str = DEFAULT_OLLAMA_MODEL_NAME, **kwargs
    ) -> BaseLanguageModel:
        logger.debug("Initializing Ollama provider")
        return OllamaLLM(model=model_name, **kwargs)

    @staticmethod
    def init_gpt4all(model_path: str | Path, **kwargs) -> BaseLanguageModel:
        logger.debug("Initializing GPT4All provider")
        if not Path(model_path).exists():
            error_msg = "Model path for GPT4All initialization does not exist"
            logger.error(error_msg)
            raise ValueError(error_msg)
        config = {**DEFAULT_OUTLINES_MODEL_PARAMETERS, **kwargs}
        return GPT4All(model=str(model_path), **config)

    @staticmethod
    def init_outlines(
        model_name: str = DEFAULT_MODEL_NAME, **kwargs
    ) -> BaseLanguageModel:
        logger.debug("Initializing Outlines provider")
        params = {**DEFAULT_OUTLINES_MODEL_PARAMETERS, **kwargs}
        return Outlines(
            model=model_name,
            **params,
        )
