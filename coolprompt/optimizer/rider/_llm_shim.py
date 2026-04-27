"""Tiny LLMClient shim that lets the unmodified RiderGenesis source from
``rider/assistant.py`` run on top of LangChain BaseLanguageModel objects
instead of OpenRouter HTTP calls.

The shim implements only the surface that RiderGenesis touches:
- ``LLMClient(provider=..., api_key=...)`` constructor signature
- ``generate(prompt=..., model=..., temperature=..., max_tokens=..., **kwargs) -> str``
- ``total_api_calls`` integer counter
- ``snapshot_generation`` / ``get_usage_stats`` no-ops (RiderGenesis only
  references these from non-LIGHT paths; kept as defensive stubs)

Routing rule: when RiderGenesis passes a model string equal to its
``PLANNING_MODEL`` class constant ("anthropic/claude-sonnet-4.6"), we
invoke the langchain ``planning_model`` registered for the active
optimizer; for any other model string we invoke the ``working_model``.
This preserves the planning/working two-tier split RiderGenesis relies
on without requiring it to know anything about langchain.

The shim is process-global by design — RiderGenesis instantiates an
LLMClient in its ``__init__``, so we register the langchain models on
the module BEFORE constructing the optimizer (see ``rider.py``).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional


_PLANNING_MODEL_NAME = "anthropic/claude-sonnet-4.6"


class _Registry:
    """Thread-local-ish registry of (working, planning) langchain models.

    LIGHT mode is single-shot per optimizer call so a simple module-level
    pair is enough; we still guard with a lock for safety if a host runs
    several RIDEROptimizer instances in parallel threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._working: Any = None
        self._planning: Any = None

    def set(self, working: Any, planning: Any) -> None:
        with self._lock:
            self._working = working
            self._planning = planning

    def get(self) -> tuple:
        with self._lock:
            return self._working, self._planning


_REGISTRY = _Registry()


def register_models(working: Any, planning: Any) -> None:
    """Register the langchain models the next LLMClient should use."""
    _REGISTRY.set(working, planning)


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
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(_coerce_to_text(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text is not None:
                    parts.append(_coerce_to_text(text))
        return "\n".join(p for p in parts if p)
    if content is not None:
        return str(content)
    return str(value)


def _invoke_with_kwargs(model: Any, prompt: str, **kwargs) -> str:
    """Try ``model.invoke(prompt, **kwargs)``, fall back to no-kwargs invoke
    for langchain backends that don't accept generation kwargs on invoke().
    """
    try:
        return _coerce_to_text(model.invoke(prompt, **kwargs))
    except TypeError:
        return _coerce_to_text(model.invoke(prompt))


class LLMClient:
    """Drop-in replacement for ``rider.llm.client.LLMClient`` that routes
    through langchain BaseLanguageModel objects registered via
    :func:`register_models`.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        max_retries: int = 1,
        retry_delay: float = 0.0,
    ) -> None:
        # Provider/api_key are accepted to match the original signature but
        # ignored — we do not perform any HTTP calls here.
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_api_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self._generation_snapshots: Dict[int, Dict[str, Any]] = {}

    # -- Core method actually used by RiderGenesis --------------------------

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
        if prompt is None and messages:
            # Flatten chat-style messages into a plain prompt — RiderGenesis
            # never uses messages in LIGHT, but be defensive.
            prompt = "\n".join(m.get("content", "") for m in messages)
        if prompt is None:
            raise ValueError("LLMClient.generate: either 'prompt' or 'messages' required")

        working, planning = _REGISTRY.get()
        if working is None:
            raise RuntimeError(
                "LLMClient shim: no langchain models registered. "
                "Call coolprompt.optimizer.rider._llm_shim.register_models(working, planning) "
                "before constructing RIDEROptimizer."
            )
        target = planning if (model == _PLANNING_MODEL_NAME and planning is not None) else working

        invoke_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens
        if top_p is not None and top_p != 1.0:
            invoke_kwargs["top_p"] = top_p

        text = _invoke_with_kwargs(target, prompt, **invoke_kwargs)
        self.total_api_calls += 1
        return text or ""

    # -- Stubs to keep RiderGenesis happy on non-LIGHT code paths -----------

    def reset_usage(self) -> None:
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._generation_snapshots = {}

    def snapshot_generation(self, generation: int) -> Dict[str, Any]:
        snap = {
            "api_calls": self.total_api_calls,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        self._generation_snapshots[generation] = snap
        return snap

    def get_generation_usage(self, generation: int) -> Dict[str, Any]:
        return self._generation_snapshots.get(generation, {
            "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        })

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_api_calls": self.total_api_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "per_generation": dict(self._generation_snapshots),
        }

    def count_tokens(self, text: str) -> int:
        # Rough char-based estimate; RiderGenesis only uses this for logging.
        return max(1, len(text) // 4)
