"""LangChain-backed LLMClient shim for the vendored RIDER Genesis source.

The vendored RIDER files are kept byte-identical. This module supplies the
``rider.llm.client.LLMClient`` surface dynamically when the copied
``assistant.py`` is loaded by the CoolPrompt wrapper.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Optional


class _ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._default_model: Any = None
        self._model_by_name: Dict[str, Any] = {}

    def set(self, default_model: Any, model_by_name: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._default_model = default_model
            self._model_by_name = dict(model_by_name or {})

    def snapshot(self) -> tuple[Any, Dict[str, Any]]:
        with self._lock:
            return self._default_model, dict(self._model_by_name)


_REGISTRY = _ModelRegistry()


def register_models(default_model: Any, model_by_name: Optional[Dict[str, Any]] = None) -> None:
    """Register LangChain models for the next RIDER Genesis instance."""

    _REGISTRY.set(default_model, model_by_name)


def _coerce_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output"):
            if key in value:
                return _coerce_to_text(value[key])
        return str(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            parts.append(_coerce_to_text(item))
        return "\n".join(part for part in parts if part)
    if content is not None:
        return str(content)
    return str(value)


def _messages_to_prompt(messages: Iterable[Dict[str, str]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _invoke_model(model: Any, prompt: str, **kwargs: Any) -> str:
    if hasattr(model, "invoke"):
        try:
            return _coerce_to_text(model.invoke(prompt, **kwargs))
        except TypeError:
            return _coerce_to_text(model.invoke(prompt))
    if callable(model):
        try:
            return _coerce_to_text(model(prompt, **kwargs))
        except TypeError:
            return _coerce_to_text(model(prompt))
    raise TypeError("Registered RIDER model must be callable or expose invoke().")


class LLMClient:
    """Drop-in subset of ``rider.llm.client.LLMClient`` backed by LangChain."""

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        max_retries: int = 1,
        retry_delay: float = 0.0,
        **_: Any,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._generation_snapshots: Dict[int, Dict[str, Any]] = {}
        self.last_response_metadata: Dict[str, Any] = {}
        self.last_error_type: Optional[str] = None
        self._default_model, self._model_by_name = _REGISTRY.snapshot()
        if self._default_model is None:
            raise RuntimeError(
                "RIDER LangChain shim has no registered model. "
                "Call register_models() before constructing RiderGenesis."
            )

    @staticmethod
    def _classify_api_exception(exc: Exception) -> str:
        text = str(exc).lower()
        if "auth" in text or "api key" in text or "401" in text:
            return "auth"
        if "not found" in text or "404" in text:
            return "not_found"
        if "context" in text and "large" in text:
            return "context_too_large"
        if "content_filter" in text or "content filter" in text or "blocked" in text:
            return "content_filter"
        if "rate" in text or "429" in text:
            return "rate_limit"
        return "exception"

    def _resolve_model(self, model_name: str) -> Any:
        return self._model_by_name.get(model_name) or self._default_model

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        if prompt is None and messages is not None:
            prompt = _messages_to_prompt(messages)
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided.")

        invoke_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p != 1.0:
            invoke_kwargs["top_p"] = top_p
        if "extra_body" in kwargs:
            invoke_kwargs["extra_body"] = kwargs["extra_body"]

        try:
            text = _invoke_model(self._resolve_model(model), prompt, **invoke_kwargs)
            self.total_api_calls += 1
            self.last_error_type = None
            self.last_response_metadata = {
                "model": model,
                "finish_reason": None,
                "completion_tokens": None,
                "max_tokens": max_tokens,
                "empty": not bool(text.strip()),
                "error_type": None,
            }
            return text
        except Exception as exc:
            self.last_error_type = self._classify_api_exception(exc)
            self.last_response_metadata = {
                "model": model,
                "finish_reason": None,
                "error_type": self.last_error_type,
                "error": str(exc)[:500],
            }
            raise

    def reset_usage(self) -> None:
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._generation_snapshots = {}

    def snapshot_generation(self, generation: int) -> Dict[str, Any]:
        snapshot = self.get_usage_stats()
        self._generation_snapshots[generation] = snapshot
        return snapshot

    def get_generation_usage(self, generation: int) -> Dict[str, Any]:
        return self._generation_snapshots.get(generation, {})

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_api_calls": self.total_api_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "per_generation": dict(self._generation_snapshots),
        }

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
