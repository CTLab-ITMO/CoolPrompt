"""Model factory for benchmark runs.

Local models are served by LM Studio's OpenAI-compatible
server (default http://localhost:1234/v1). Load the target
model in LM Studio first (`lms load <id>`), then pass its
ladder key or API id via make_llm().

A cross-family Anthropic slot is available when CP_ANTHROPIC_KEY
is set; it is part of the thesis design but execution is
local-only for now.

Model API ids below were captured from `GET /v1/models` on this
machine (they match the `lms ls` identifiers).
"""

import os

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

LMSTUDIO_BASE_URL = os.environ.get(
    "CP_LMSTUDIO_URL", "http://localhost:1234/v1"
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

# Cross-family API slot (design only; needs CP_ANTHROPIC_KEY).
ANTHROPIC_MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
}


def make_llm(
    name: str,
    temperature: float = 0.0,
    max_retries: int = 10,
    request_timeout: int = 600,
) -> BaseLanguageModel:
    """Build a LangChain chat model for a benchmark run.

    Args:
        name: A MODEL_LADDER key, an ANTHROPIC_MODELS key, or a
            raw LM Studio API model id.
        temperature: Sampling temperature.
        max_retries: Client-side retry count.
        request_timeout: Per-request timeout (seconds); local
            models can be slow, so the default is generous.

    Returns:
        A configured BaseLanguageModel.
    """
    if name in ANTHROPIC_MODELS:
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("CP_ANTHROPIC_KEY")
        if not api_key:
            raise RuntimeError(
                "CP_ANTHROPIC_KEY not set for Anthropic model "
                f"'{name}'"
            )
        return ChatAnthropic(
            model=ANTHROPIC_MODELS[name],
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=request_timeout,
        )

    model_id = MODEL_LADDER.get(name, name)
    return ChatOpenAI(
        model=model_id,
        base_url=LMSTUDIO_BASE_URL,
        api_key=os.environ.get("CP_LMSTUDIO_KEY", "lm-studio"),
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )
