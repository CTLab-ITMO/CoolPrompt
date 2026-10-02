"""Validation orchestration for generated examples."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coolprompt.spec_generator.models import Example, GenerationContext
from coolprompt.spec_generator.validation.format import Deduplicator, ExampleValidator
from coolprompt.utils.logging_config import logger

Producer = Callable[[int], list[Any]]


class ValidationPipeline:
    """Validate, deduplicate, and top up examples."""

    def __init__(
        self,
        validator: ExampleValidator,
        deduplicator: Deduplicator,
        *,
        max_topup_attempts: int = 10,
    ) -> None:
        """Initialize validation components and the top-up attempt limit."""

        if max_topup_attempts < 1:
            raise ValueError("max_topup_attempts must be at least 1")

        self._validator = validator
        self._deduplicator = deduplicator
        self._max_topup_attempts = max_topup_attempts

    def run(
        self,
        producer: Producer,
        context: GenerationContext,
        target_n: int,
        *,
        reset_deduplicator: bool = True,
        accept_candidate: Callable[[Example], bool] | None = None,
        on_accept: Callable[[Example], None] | None = None,
    ) -> list[Example]:
        """Produce, validate, deduplicate, and top up to the target size."""

        if target_n < 0:
            raise ValueError("target_n must be non-negative")
        if target_n == 0:
            return []

        if reset_deduplicator:
            self._deduplicator.reset()

        accepted: list[Example] = []

        for attempt in range(1, self._max_topup_attempts + 1):
            remaining = target_n - len(accepted)
            if remaining <= 0:
                break

            raw = producer(remaining)
            if not raw:
                logger.warning(
                    "Validation round %d/%d produced no examples.",
                    attempt,
                    self._max_topup_attempts,
                )
                continue

            valid, invalid = self._validator.validate(raw, context.spec)

            new: list[Example] = []

            for example in valid:
                if len(new) >= remaining:
                    break

                if accept_candidate is not None and not accept_candidate(example):
                    continue

                if not self._deduplicator.accept(example):
                    continue

                new.append(example)

                if on_accept is not None:
                    on_accept(example)

            accepted.extend(new)

            logger.info(
                "Validation round %d/%d: raw=%d invalid=%d accepted=%d total=%d/%d",
                attempt,
                self._max_topup_attempts,
                len(raw),
                len(invalid),
                len(new),
                len(accepted),
                target_n,
            )

        if len(accepted) < target_n:
            logger.warning(
                "Validation stopped with %d/%d accepted examples.",
                len(accepted),
                target_n,
            )

        return accepted
