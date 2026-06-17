"""Runtime settings for the CoolPrompt demo service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_api_key() -> str | None:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("RIDER_KEY_1")
    )


def _looks_like_openrouter_key(api_key: str | None) -> bool:
    return bool(api_key and api_key.strip().startswith("sk-or-"))


def _env_base_url() -> str | None:
    configured = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if configured:
        return configured
    if _looks_like_openrouter_key(_env_api_key()):
        return OPENROUTER_BASE_URL
    return None


def _env_model_name() -> str:
    configured = os.getenv("COOLPROMPT_DEMO_MODEL")
    if configured:
        return configured
    if _looks_like_openrouter_key(_env_api_key()):
        return "google/gemini-2.5-flash"
    return "gpt-4o-mini"


@dataclass(frozen=True)
class DemoSettings:
    """Environment-driven service settings."""

    app_name: str = "CoolPrompt Interface Demo"
    model_name: str = field(default_factory=_env_model_name)
    openai_api_key: str | None = field(default_factory=_env_api_key)
    openai_base_url: str | None = field(default_factory=_env_base_url)
    allow_mock: bool = _bool_env("COOLPROMPT_DEMO_ALLOW_MOCK", default=False)
    force_mock: bool = _bool_env("COOLPROMPT_DEMO_MOCK", default=False)
    max_compare_methods: int = int(os.getenv("COOLPROMPT_MAX_COMPARE_METHODS", "4"))
    max_workers: int = int(os.getenv("COOLPROMPT_DEMO_WORKERS", "2"))
    max_compare_workers: int = int(os.getenv("COOLPROMPT_COMPARE_WORKERS", "1"))
    request_timeout_seconds: int = int(os.getenv("COOLPROMPT_DEMO_TIMEOUT_SECONDS", "900"))
    lightweight_hyper_similarity: bool = _bool_env(
        "COOLPROMPT_DEMO_LIGHTWEIGHT_HYPER_MMR",
        default=True,
    )

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> DemoSettings:
    """Return current settings."""

    return DemoSettings()
