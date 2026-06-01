"""Model factory for benchmark runs.

Two backends, both OpenAI-compatible, selected by `backend`
(or the CP_BACKEND env var; default "lmstudio"):

- "lmstudio": local models served by LM Studio at
  http://localhost:1234/v1 (override via CP_LMSTUDIO_URL).
  Load the model first (`lms load <id>`). Free but slow, and
  the MLX runtime is fragile under heavy structured-output
  methods (pe2_sgr).
- "openrouter": cloud models via https://openrouter.ai/api/v1
  (needs CP_OPENROUTER_KEY). Fast, reliable structured output,
  and safe to run with several --workers in parallel.

A cross-family Anthropic slot is available when CP_ANTHROPIC_KEY
is set.

LM Studio model ids below were captured from `GET /v1/models`
on this machine (they match the `lms ls` identifiers).
"""

import os

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

LMSTUDIO_BASE_URL = os.environ.get(
    "CP_LMSTUDIO_URL", "http://localhost:1234/v1"
)
OPENROUTER_BASE_URL = os.environ.get(
    "CP_OPENROUTER_URL", "https://openrouter.ai/api/v1"
)

# Capability ladder: logical name -> LM Studio API model id.
MODEL_LADDER = {
    "weak": "qwen/qwen3-1.7b",
    "mid": "qwen3-4b-instruct-2507-mlx",
    "strong": "qwen/qwen3-14b",
    # Cross-family local check (not Qwen): OpenAI gpt-oss 20B.
    "cross": "openai/gpt-oss-20b",
    # Capable local judge for llm_as_judge metrics.
    "judge": "qwen3-30b-a3b-instruct-2507-mlx",
}

# OpenRouter ladder: logical name -> OpenRouter model slug.
# Mirrors the Qwen3 scaling study; cross-family + judge use
# flagships with reliable structured-output support.
OPENROUTER_LADDER = {
    "weak": "qwen/qwen3-8b",
    "mid": "qwen/qwen3-14b",
    "strong": "qwen/qwen3-30b-a3b-instruct-2507",
    "cross": "openai/gpt-4o-mini",
    "judge": "qwen/qwen3-235b-a22b-2507",
}

# Native OpenAI ladder: logical name -> OpenAI model id
# (backend "openai", needs CP_OPENAI_KEY).
OPENAI_MODELS = {
    "cross": "gpt-4o-mini",
}

# Cross-family API slot (needs CP_ANTHROPIC_KEY).
ANTHROPIC_MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
}


def make_llm(
    name: str,
    backend: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 10,
    request_timeout: int = 600,
) -> BaseLanguageModel:
    """Build a LangChain chat model for a benchmark run.

    Args:
        name: A ladder key ("weak"/"mid"/"strong"/"cross"/
            "judge"), an ANTHROPIC_MODELS/OPENAI_MODELS key, or a
            raw model id for the resolved backend.
        backend: Force a backend ("lmstudio", "openrouter",
            "openai", "anthropic"). If omitted, falls back to the
            CP_BACKEND env var, then AUTO-ROUTES by model
            identity: OpenAI models -> native OpenAI, Anthropic
            models -> native Anthropic, everything else ->
            OpenRouter. (lmstudio is never auto-selected; request
            it explicitly for local runs.)
        temperature: Sampling temperature.
        max_retries: Client-side retry count.
        request_timeout: Per-request timeout (seconds).

    Returns:
        A configured BaseLanguageModel.
    """
    backend = backend or os.environ.get("CP_BACKEND")
    if backend is None:
        if name in ANTHROPIC_MODELS or name.startswith("claude"):
            backend = "anthropic"
        elif name in OPENAI_MODELS or name.startswith("gpt-"):
            backend = "openai"
        else:
            backend = "openrouter"

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("CP_ANTHROPIC_KEY")
        if not api_key:
            raise RuntimeError(
                "CP_ANTHROPIC_KEY not set for Anthropic model "
                f"'{name}'"
            )
        return ChatAnthropic(
            model=ANTHROPIC_MODELS.get(name, name),
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=request_timeout,
        )

    if backend == "openai":
        api_key = os.environ.get("CP_OPENAI_KEY")
        if not api_key:
            raise RuntimeError(
                "CP_OPENAI_KEY not set for openai backend"
            )
        return ChatOpenAI(
            model=OPENAI_MODELS.get(name, name),
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

    if backend == "openrouter":
        api_key = os.environ.get("CP_OPENROUTER_KEY")
        if not api_key:
            raise RuntimeError(
                "CP_OPENROUTER_KEY not set for openrouter backend"
            )
        return ChatOpenAI(
            model=OPENROUTER_LADDER.get(name, name),
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

    return ChatOpenAI(
        model=MODEL_LADDER.get(name, name),
        base_url=LMSTUDIO_BASE_URL,
        api_key=os.environ.get("CP_LMSTUDIO_KEY", "lm-studio"),
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )
