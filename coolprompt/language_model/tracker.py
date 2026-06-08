from langchain_community.callbacks import get_openai_callback

from typing import Any
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

import time


class OpenAITracker:
    """Tracks OpenAI API usage stats like tokens and costs.

    Keeps single instance to collect stats from all model calls.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset_stats()
        return cls._instance

    def _reset_stats(self):
        """Resets all tracking stats to zero."""
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "invoke_calls": 0,
            "batch_calls": 0,
            "batch_items": 0,
            "api_wait_sec": 0.0,
        }

    def _update_stats(self, callback, invoke_flag, batch_size, duration_sec, **kwargs):
        """Updates stats from callback data.

        Args:
            callback: OpenAI callback with token and cost info.
            invoke_flag (bool): True for invoke calls, False for batch.
            batch_size (int): Number of items in batch call.
        """
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += callback.total_tokens
        self.stats["prompt_tokens"] += callback.prompt_tokens
        self.stats["completion_tokens"] += callback.completion_tokens
        self.stats["total_cost"] += callback.total_cost
        self.stats["api_wait_sec"] += duration_sec

        if invoke_flag:
            self.stats["invoke_calls"] += 1
        else:
            self.stats["batch_calls"] += 1
            self.stats["batch_items"] += batch_size

    def wrap_model(self, model):
        """Wraps model with tracking wrapper.

        Args:
            model: LangChain language model to wrap.

        Returns:
            TrackedLLMWrapper: Model wrapper with tracking.
        """
        return TrackedLLMWrapper(model, self)

    def get_stats(self):
        """Gets current tracking stats.

        Returns:
            dict: Copy of current stats dictionary.
        """
        return self.stats.copy()

    def reset_stats(self):
        """Resets all tracking stats to zero."""
        self._reset_stats()


class TrackedLLMWrapper(BaseLanguageModel):
    """Wrapper for LangChain models that tracks API usage.

    Tracks tokens, costs, and API wait time for all invoke and batch calls.
    """

    model: Any
    tracker: Any

    def __init__(self, model, tracker):
        super().__init__(model=model, tracker=tracker)

    @property
    def _llm_type(self):
        return "tracked_" + getattr(self.model, "_llm_type", "llm")

    def generate_prompt(self, prompts, stop=None, **kwargs):
        return self.model.generate_prompt(prompts, stop=stop, **kwargs)

    async def agenerate_prompt(self, prompts, stop=None, **kwargs):
        return await self.model.agenerate_prompt(prompts, stop=stop, **kwargs)

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        """Calls model and tracks usage stats.

        Args:
            input: Input to pass to model.
            config: Optional LangChain runnable config.
            stop: Optional stop sequences.
            **kwargs: Additional model arguments.

        Returns:
            Model output.
        """
        start_time = time.time()
        with get_openai_callback() as cb:
            result = self.model.invoke(
                input, config=config, stop=stop, config=config, **kwargs
            )
        duration_sec = time.time() - start_time
        self.tracker._update_stats(cb, invoke_flag=True, batch_size=0, duration_sec=duration_sec)
        return result

    def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        """Calls model in batch and tracks usage stats.

        Args:
            inputs: List of inputs to process.
            config: Optional LangChain runnable config.
            return_exceptions: Whether to return exceptions instead of raising.
            **kwargs: Additional model arguments.

        Returns:
            List of model outputs.
        """
        start_time = time.time()
        with get_openai_callback() as cb:
            results = self.model.batch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                config=config, **kwargs,
            )
        duration_sec = time.time() - start_time
        self.tracker._update_stats(cb, invoke_flag=False, batch_size=len(inputs), duration_sec=duration_sec)
        return results

    def with_structured_output(self, schema, **kwargs):
        """Returns model with structured output support.

        Args:
            schema: Output schema to use.
            **kwargs: Additional arguments.

        Returns:
            Model with structured output.

        Raises:
            NotImplementedError: If model does not support structured output.
        """
        if hasattr(self.model, "with_structured_output"):
            return model_tracker.wrap_model(self.model.with_structured_output(schema, **kwargs))
        raise NotImplementedError(
            f"Model {type(self.model)} does not support structured output"
        )

    def reset_stats(self):
        """Resets all tracking stats to zero."""
        self.tracker.reset_stats()

    def get_stats(self):
        """Gets current tracking stats.

        Returns:
            dict: Copy of current stats dictionary.
        """
        return self.tracker.get_stats()

    def __getattr__(self, name):
        return getattr(self.model, name)


model_tracker = OpenAITracker()


def create_chat_model(model=None, **kwargs):
    """Creates ChatOpenAI model wrapped with tracking.

    Args:
        model: Optional existing model or model name string.
        **kwargs: ChatOpenAI initialization arguments.

    Returns:
        TrackedLLMWrapper: Model wrapped with usage tracking.
    """
    if isinstance(model, BaseLanguageModel):
        base_model = model
    elif model is not None:
        kwargs["model"] = model
        base_model = ChatOpenAI(**kwargs)
    else:
        base_model = ChatOpenAI(**kwargs)

    return model_tracker.wrap_model(base_model)
