"""LangChain-backed LLMClient shim for RIDER Genesis Ultra.

This module supplies the ``rider.llm.client.LLMClient`` surface dynamically
when the RIDER core is loaded by the CoolPrompt wrapper.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Iterable, List, Optional, Type

from pydantic import BaseModel


class _ModelRegistry:
    """Thread-safe registry used while a RIDER instance is built."""

    def __init__(self) -> None:
        """Initialize an empty model registry."""

        self._lock = threading.Lock()
        self._default_model: Any = None
        self._model_by_name: Dict[str, Any] = {}

    def set(
        self,
        default_model: Any,
        model_by_name: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store default and alias-specific LangChain models.

        Args:
            default_model: Fallback model used when RIDER requests an unknown
                alias.
            model_by_name: Optional mapping from RIDER model names to
                LangChain model instances.
        """

        with self._lock:
            self._default_model = default_model
            self._model_by_name = dict(model_by_name or {})

    def snapshot(self) -> tuple[Any, Dict[str, Any]]:
        """Return an immutable view of the current registry.

        Returns:
            Tuple of default model and a shallow copy of alias mapping.
        """

        with self._lock:
            return self._default_model, dict(self._model_by_name)


_REGISTRY = _ModelRegistry()


def register_models(
    default_model: Any,
    model_by_name: Optional[Dict[str, Any]] = None,
) -> None:
    """Register LangChain models for the next RIDER Genesis instance.

    Args:
        default_model: Fallback LangChain model for unresolved RIDER aliases.
        model_by_name: Optional mapping from RIDER model names to
            LangChain model instances.
    """

    _REGISTRY.set(default_model, model_by_name)


def _coerce_to_text(value: Any) -> str:
    """Convert common LangChain response shapes into plain text.

    Args:
        value: Raw model response, message, dict, or primitive value.

    Returns:
        Text content suitable for the RIDER runtime.
    """

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
    """Flatten chat messages into the prompt string expected by LangChain LLMs.

    Args:
        messages: Iterable of OpenAI-style message dictionaries.

    Returns:
        Newline-joined message contents.
    """

    return "\n".join(str(message.get("content", "")) for message in messages)


def _invoke_model(model: Any, prompt: str, **kwargs: Any) -> str:
    """Invoke a LangChain model or callable and normalize its response.

    Args:
        model: LangChain model-like object exposing ``invoke`` or a callable.
        prompt: Prompt text to send.
        **kwargs: Optional generation parameters.

    Returns:
        Response text.

    Raises:
        TypeError: If the registered model cannot be invoked.
    """

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
    raise TypeError(
        "Registered RIDER model must be callable or expose invoke()."
    )


def _coerce_to_schema(value: Any, schema: Type[BaseModel]) -> BaseModel:
    """Convert common structured-output shapes into a Pydantic object."""

    if isinstance(value, schema):
        return value
    if isinstance(value, BaseModel):
        return schema.model_validate(value.model_dump())
    if isinstance(value, dict):
        return schema.model_validate(value)
    if isinstance(value, str):
        return schema.model_validate(json.loads(value))
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return schema.model_validate(json.loads(content))
    if isinstance(content, dict):
        return schema.model_validate(content)
    return schema.model_validate(value)


def _with_structured_output(model: Any, schema: Type[BaseModel]) -> Any:
    """Return a LangChain structured-output wrapper for a model."""

    if not hasattr(model, "with_structured_output"):
        raise NotImplementedError(
            "Registered RIDER model does not expose with_structured_output()."
        )
    try:
        return model.with_structured_output(schema=schema, method="json_schema")
    except TypeError:
        try:
            return model.with_structured_output(schema, method="json_schema")
        except TypeError:
            return model.with_structured_output(schema)


def _invoke_structured_model(
    model: Any,
    prompt: str,
    schema: Type[BaseModel],
    **kwargs: Any,
) -> BaseModel:
    """Invoke a LangChain structured-output model and validate its result."""

    structured_model = _with_structured_output(model, schema)
    if not hasattr(structured_model, "invoke"):
        raise TypeError("Structured RIDER model must expose invoke().")
    try:
        value = structured_model.invoke(prompt, **kwargs)
    except TypeError:
        value = structured_model.invoke(prompt)
    return _coerce_to_schema(value, schema)


class _LangChainStructuredClient:
    """Instructor-like adapter backed by LangChain structured output."""

    def __init__(self, llm_client: "LLMClient", model_name: str) -> None:
        self._llm_client = llm_client
        self._model_name = model_name

    def create(
        self,
        *,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> BaseModel:
        """Create a Pydantic response via ``with_structured_output``.

        The signature intentionally mirrors the subset of Instructor used by
        the copied RIDER core, so the core can keep one structured-call path.
        """

        prompt = _messages_to_prompt(messages)
        invoke_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "extra_body" in kwargs:
            invoke_kwargs["extra_body"] = kwargs["extra_body"]

        attempts = max(1, int(max_retries or 1))
        last_exc: Optional[Exception] = None
        for _ in range(attempts):
            try:
                obj = _invoke_structured_model(
                    self._llm_client._resolve_model(self._model_name),
                    prompt,
                    response_model,
                    **invoke_kwargs,
                )
                self._llm_client.total_api_calls += 1
                self._llm_client.last_error_type = None
                self._llm_client.last_response_metadata = {
                    "model": self._model_name,
                    "finish_reason": None,
                    "completion_tokens": None,
                    "max_tokens": max_tokens,
                    "empty": False,
                    "error_type": None,
                    "structured": True,
                }
                return obj
            except Exception as exc:
                last_exc = exc
                self._llm_client.last_error_type = (
                    self._llm_client._classify_api_exception(exc)
                )
                self._llm_client.last_response_metadata = {
                    "model": self._model_name,
                    "finish_reason": None,
                    "error_type": self._llm_client.last_error_type,
                    "error": str(exc)[:500],
                    "structured": True,
                }
        assert last_exc is not None
        raise last_exc


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
        """Create a LangChain-backed client with the RIDER interface.

        Args:
            provider: Provider name kept for RIDER compatibility.
            api_key: API key value kept for RIDER compatibility. The shim
                routes calls through registered LangChain models instead.
            max_retries: Retry count field kept for compatibility.
            retry_delay: Retry delay field kept for compatibility.
            **_: Extra RIDER options ignored by the shim.

        Raises:
            RuntimeError: If no LangChain model was registered before
                construction.
        """

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
        """Classify model invocation errors using RIDER-compatible labels.

        Args:
            exc: Exception raised by the registered model.

        Returns:
            RIDER diagnostic label such as ``"auth"`` or ``"rate_limit"``.
        """

        text = str(exc).lower()
        if "auth" in text or "api key" in text or "401" in text:
            return "auth"
        if "not found" in text or "404" in text:
            return "not_found"
        if "context" in text and "large" in text:
            return "context_too_large"
        if (
            "content_filter" in text
            or "content filter" in text
            or "blocked" in text
        ):
            return "content_filter"
        if "rate" in text or "429" in text:
            return "rate_limit"
        return "exception"

    def _resolve_model(self, model_name: str) -> Any:
        """Resolve a RIDER model alias to a LangChain model.

        Args:
            model_name: Model name requested by RIDER.

        Returns:
            Registered model for the alias, or the default model.
        """

        return self._model_by_name.get(model_name) or self._default_model

    def structured_client(self, model_name: str) -> _LangChainStructuredClient:
        """Return an Instructor-compatible client for structured RIDER calls."""

        return _LangChainStructuredClient(self, model_name)

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
        """Generate text through the registered LangChain model.

        Args:
            prompt: Plain prompt string.
            messages: Optional chat messages used when ``prompt`` is absent.
            model: RIDER model alias to resolve.
            temperature: Sampling temperature forwarded when supported.
            max_tokens: Maximum output tokens forwarded when supported.
            top_p: Nucleus sampling value forwarded when not default.
            **kwargs: Optional extra generation arguments; currently
                ``extra_body`` is forwarded for OpenRouter-style clients.

        Returns:
            Generated text.

        Raises:
            ValueError: If neither ``prompt`` nor ``messages`` is provided.
            Exception: Re-raises model invocation errors after recording RIDER
                diagnostic metadata.
        """

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
            text = _invoke_model(
                self._resolve_model(model),
                prompt,
                **invoke_kwargs,
            )
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
        """Reset RIDER-compatible usage counters."""

        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._generation_snapshots = {}

    def snapshot_generation(self, generation: int) -> Dict[str, Any]:
        """Store current usage statistics for a RIDER generation.

        Args:
            generation: Generation index reported by RIDER.

        Returns:
            Snapshot dictionary from ``get_usage_stats``.
        """

        snapshot = self.get_usage_stats()
        self._generation_snapshots[generation] = snapshot
        return snapshot

    def get_generation_usage(self, generation: int) -> Dict[str, Any]:
        """Return a previously stored generation usage snapshot.

        Args:
            generation: Generation index.

        Returns:
            Snapshot dictionary, or an empty dict when absent.
        """

        return self._generation_snapshots.get(generation, {})

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return RIDER-compatible aggregate usage statistics.

        Returns:
            Usage counters and per-generation snapshots.
        """

        return {
            "total_api_calls": self.total_api_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "per_generation": dict(self._generation_snapshots),
        }

    def count_tokens(self, text: str) -> int:
        """Estimate token count for RIDER budget logic.

        Args:
            text: Text to estimate.

        Returns:
            Rough token count using a four-character heuristic.
        """

        return max(1, len(text) // 4)
