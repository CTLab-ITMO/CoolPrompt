"""Runtime settings for the CoolPrompt demo service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DemoSettings:
    """Environment-driven service settings."""

    app_name: str = "CoolPrompt Interface Demo"
    model_name: str = os.getenv("COOLPROMPT_DEMO_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    allow_mock: bool = _bool_env("COOLPROMPT_DEMO_ALLOW_MOCK", default=False)
    force_mock: bool = _bool_env("COOLPROMPT_DEMO_MOCK", default=False)
    max_compare_methods: int = int(os.getenv("COOLPROMPT_MAX_COMPARE_METHODS", "4"))
    max_workers: int = int(os.getenv("COOLPROMPT_DEMO_WORKERS", "2"))
    request_timeout_seconds: int = int(os.getenv("COOLPROMPT_DEMO_TIMEOUT_SECONDS", "900"))

    @property
    def has_openai_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


def get_settings() -> DemoSettings:
    """Return current settings."""

    return DemoSettings()
