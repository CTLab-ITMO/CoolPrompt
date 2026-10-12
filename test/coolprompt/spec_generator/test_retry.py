import pytest

from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"min_wait_seconds": -1},
        {"max_wait_seconds": float("inf")},
        {"min_wait_seconds": 2, "max_wait_seconds": 1},
    ),
)
def test_retry_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        RetryConfig(**kwargs)


def test_retry_retries_transient_errors_with_exponential_backoff(monkeypatch) -> None:
    attempts = 0
    waits: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    monkeypatch.setattr(
        "coolprompt.spec_generator.utils.retry.time.sleep",
        waits.append,
    )

    result = invoke_with_retry(
        operation,
        RetryConfig(max_retries=3, min_wait_seconds=0.5, max_wait_seconds=1),
    )

    assert result == "ok"
    assert attempts == 3
    assert waits == [0.5, 1]


def test_retry_raises_after_budget_is_exhausted(monkeypatch) -> None:
    attempts = 0
    monkeypatch.setattr(
        "coolprompt.spec_generator.utils.retry.time.sleep",
        lambda _: None,
    )

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("still unavailable")

    with pytest.raises(ConnectionError, match="still unavailable"):
        invoke_with_retry(
            operation,
            RetryConfig(max_retries=2, min_wait_seconds=0, max_wait_seconds=0),
        )

    assert attempts == 3


def test_retry_does_not_retry_unlisted_errors() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        invoke_with_retry(operation, RetryConfig(max_retries=3))

    assert attempts == 1
