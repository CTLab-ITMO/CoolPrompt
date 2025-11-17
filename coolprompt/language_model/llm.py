"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

import functools
import threading
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.language_models.base import BaseLanguageModel
from langchain_community.callbacks.manager import get_openai_callback
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
            pipeline_kwargs=generation_and_model_config,
        )
        return ChatHuggingFace(llm=llm)


class TrackedLLMWrapper:
    """Простая обертка вокруг ChatOpenAI с трекингом"""

    def __init__(self, model, tracker):
        self.model = model
        self.tracker = tracker

    def invoke(self, input, **kwargs):
        with get_openai_callback() as cb:
            result = self.model.invoke(input, **kwargs)
            self.tracker._update_stats(cb, True)
            return result

    def batch(self, inputs, **kwargs):
        with get_openai_callback() as cb:
            results = self.model.batch(inputs, **kwargs)
            self.tracker._update_stats(cb, False, batch_size=len(inputs))
            return results

    def reset_stats(self):
        self.tracker.reset_stats()

    def get_stats(self):
        return self.tracker.get_stats()

    # Проксируем остальные методы
    def __getattr__(self, name):
        return getattr(self.model, name)


class OpenAITracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset_stats()
        return cls._instance

    def _reset_stats(self):
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "invoke_calls": 0,
            "batch_calls": 0,
            "batch_items": 0,
        }

    def _update_stats(self, callback, invoke_flag, **kwargs):
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += callback.total_tokens
        self.stats["prompt_tokens"] += callback.prompt_tokens
        self.stats["completion_tokens"] += callback.completion_tokens
        self.stats["total_cost"] += callback.total_cost
        if invoke_flag:
            self.stats["invoke_calls"] += 1
        else:
            self.stats["batch_calls"] += 1
            self.stats["batch_items"] += kwargs.get("batch_size", 0)

    def wrap_model(self, model):
        """Обертывает модель для трекинга"""
        return TrackedLLMWrapper(model, self)

    def get_stats(self):
        return self.stats.copy()

    def reset_stats(self):
        self._reset_stats()
