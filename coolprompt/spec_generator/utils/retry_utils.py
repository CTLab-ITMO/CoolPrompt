from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

_TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
)


@dataclass(frozen=True)
class RetryConfig:
    max_network_retries: int = 3
    network_retry_min_wait: float = 2.0
    network_retry_max_wait: float = 20.0

    def __post_init__(self) -> None:
        if self.max_network_retries < 0:
            raise ValueError(
                "max_network_retries must be greater than or equal to 0"
            )

        if self.network_retry_min_wait < 0:
            raise ValueError(
                "network_retry_min_wait must be greater than or equal to 0"
            )

        if self.network_retry_max_wait < 0:
            raise ValueError(
                "network_retry_max_wait must be greater than or equal to 0"
            )

        if self.network_retry_min_wait > self.network_retry_max_wait:
            raise ValueError(
                "network_retry_min_wait must be less than or equal to "
                "network_retry_max_wait"
            )


def invoke_with_retry(
        operation: Callable[[], T],
        config: RetryConfig,
) -> T:
    retrying = retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(
            min=config.network_retry_min_wait,
            max=config.network_retry_max_wait,
        ),
        stop=stop_after_attempt(
            config.max_network_retries + 1
        ),
        reraise=True,
    )(operation)

    return retrying()
