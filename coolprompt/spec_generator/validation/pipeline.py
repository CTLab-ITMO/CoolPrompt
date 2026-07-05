from __future__ import annotations

from typing import Any, Callable

from coolprompt.spec_generator.schema import TaskSpec
from coolprompt.spec_generator.utils.retry_config import ValidationConfig
from coolprompt.spec_generator.validation.example_models import ExampleBase
from coolprompt.spec_generator.validation.format_validator import (
    Deduplicator,
    FormatValidator,
)
from coolprompt.spec_generator.validation.judge import LLMJudge
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger

RawBatchProducer = Callable[[int], list[Any]]


class ValidationPipeline:
    def __init__(
            self,
            format_validator: FormatValidator,
            deduplicator: Deduplicator,
            judge: LLMJudge,
            config: ValidationConfig | None = None,
    ) -> None:
        self._format_validator = format_validator
        self._deduplicator = deduplicator
        self._judge = judge
        self._config = config or ValidationConfig()

    def run(
            self,
            raw_batch_producer: RawBatchProducer,
            spec: TaskSpec,
            task: Task,
            target_n: int,
            is_corner: bool = False,
    ) -> list[ExampleBase]:
        if target_n <= 0:
            return []

        dataset: list[ExampleBase] = []

        for attempt in range(1, self._config.max_topup_attempts + 1):
            remaining = target_n - len(dataset)

            if remaining <= 0:
                break

            raw = raw_batch_producer(remaining)

            if not raw:
                logger.warning(
                    "Round %d/%d produced no raw examples.",
                    attempt,
                    self._config.max_topup_attempts,
                )
                continue

            valid, invalid = self._format_validator.validate(raw, spec, task)
            valid = (self._deduplicator.dedupe_exact_pairs_within_batch(valid))

            if self._config.judge_enabled:
                valid, rejected = self._judge.filter(valid, spec, is_corner=is_corner)
            else:
                rejected = []

            accepted = self._deduplicator.filter(valid, limit=remaining)
            dataset.extend(accepted)

            logger.info(
                "Round %d/%d: raw=%d, structural_invalid=%d, "
                "judge_rejected=%d, accepted=%d, total=%d/%d",
                attempt,
                self._config.max_topup_attempts,
                len(raw),
                len(invalid),
                len(rejected),
                len(accepted),
                len(dataset),
                target_n,
            )

        if len(dataset) < target_n:
            logger.warning("Stopped with %d/%d examples.", len(dataset), target_n)

        return dataset
