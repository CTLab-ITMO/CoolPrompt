"""Retry helpers for transient model-call failures."""

from __future__ import annotations

import time
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry policy for model calls."""

    max_retries: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 8.0

    def __post_init__(self) -> None:
        """Validate retry counts and backoff bounds."""

        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if any(
            not math.isfinite(wait) or wait < 0
            for wait in (self.min_wait_seconds, self.max_wait_seconds)
        ):
            raise ValueError("retry waits must be finite and non-negative")

        if self.min_wait_seconds > self.max_wait_seconds:
            raise ValueError("min_wait_seconds must not exceed max_wait_seconds")


def invoke_with_retry(
    operation: Callable[[], T],
    config: RetryConfig,
    *,
    extra_retry_exceptions: tuple[type[Exception], ...] = (),
) -> T:
    """Run ``operation`` with exponential backoff for retryable exceptions."""

    retryable = _TRANSIENT_ERRORS + extra_retry_exceptions
    delay = config.min_wait_seconds

    for attempt in range(config.max_retries + 1):
        try:
            return operation()
        except retryable:
            if attempt == config.max_retries:
                raise

            time.sleep(delay)
            delay = min(config.max_wait_seconds, delay * 2)

    raise RuntimeError("unreachable retry state")
