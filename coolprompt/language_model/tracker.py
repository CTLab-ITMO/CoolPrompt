from langchain_community.callbacks import get_openai_callback

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import Any


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
        return TrackedLLMWrapper(model, self)

    def get_stats(self):
        return self.stats.copy()

    def reset_stats(self):
        self._reset_stats()


class TrackedLLMWrapper(BaseLanguageModel):
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

    def with_structured_output(self, schema, **kwargs):
        if hasattr(self.model, 'with_structured_output'):
            return self.model.with_structured_output(schema, **kwargs)
        raise NotImplementedError(
            f"Model {type(self.model)} does not support structured output"
        )

    def reset_stats(self):
        self.tracker.reset_stats()

    def get_stats(self):
        return self.tracker.get_stats()

    def __getattr__(self, name):
        return getattr(self.model, name)


model_tracker = OpenAITracker()


def create_chat_model(model=None, **kwargs):
    if isinstance(model, BaseLanguageModel):
        base_model = model
    elif model is not None:
        kwargs["model"] = model
        base_model = ChatOpenAI(**kwargs)
    else:
        base_model = ChatOpenAI(**kwargs)
    
    return model_tracker.wrap_model(base_model)

