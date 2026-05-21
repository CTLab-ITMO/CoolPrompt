"""Integration tests: real ChatOpenAI calls via OpenRouter (or compatible base_url).

Set one of:
  - ``OPENROUTER_API_KEY`` (preferred), or ``OPENAI_API_KEY`` (same value works for OpenRouter),
  optional ``OPENROUTER_BASE_URL`` (default ``https://openrouter.ai/api/v1``),
  optional ``INTEGRATION_LLM_MODEL`` (default ``openai/gpt-4o-mini``).

Example::

    export OPENROUTER_API_KEY=sk-or-v1-...
    export INTEGRATION_LLM_MODEL=openai/gpt-5-nano
    pytest test/coolprompt/optimizer/hyper/ -m integration -q
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _integration_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _integration_chat_model() -> Any:
    from langchain_openai import ChatOpenAI

    api_key = _integration_api_key()
    if not api_key:
        raise RuntimeError("integration API key missing")
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    model = os.environ.get("INTEGRATION_LLM_MODEL", "openai/gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=0.7,
        max_completion_tokens=512,
        max_retries=5,
        base_url=base_url,
        api_key=api_key,
    )


skip_no_router_key = pytest.mark.skipif(
    not _integration_api_key(),
    reason="Set OPENROUTER_API_KEY or OPENAI_API_KEY for OpenRouter ChatOpenAI",
)


@skip_no_router_key
def test_meta_prompt_optimizer_real_llm_smoke():
    from coolprompt.optimizer.hyper.meta_prompt import MetaPromptOptimizer

    llm = _integration_chat_model()
    opt = MetaPromptOptimizer(model=llm)
    out = opt.optimize(
        "Summarize the following in one sentence: hello world.",
        meta_info={"problem_description": "toy summarization"},
    )
    assert isinstance(out, str)
    assert len(out) > 5


@skip_no_router_key
def test_hyper_light_method_real_llm_smoke():
    from coolprompt.optimizer.hyper.meta_prompt import HyPERLightMethod

    llm = _integration_chat_model()
    method = HyPERLightMethod()
    out = method.optimize(
        model=llm,
        initial_prompt="Translate to French: good morning",
        problem_description="translation",
        meta_prompt_context={"style": "informal"},
    )
    assert isinstance(out, str)
    assert len(out) > 3
