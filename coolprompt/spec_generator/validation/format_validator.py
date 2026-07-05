from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ValidationError

from coolprompt.spec_generator.schema import TaskSpec
from coolprompt.spec_generator.validation.example_models import (
    ExampleBase,
    build_example_model,
)
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.strip().casefold().split())


def _shingles(normalized_text: str, size: int) -> set[str]:
    words = _WORD_RE.findall(normalized_text)

    if not words:
        return set()

    if len(words) < size:
        return {" ".join(words)}

    return {
        " ".join(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0

    union_size = len(left | right)

    if union_size == 0:
        return 0.0

    return len(left & right) / union_size


def _model_cache_key(spec: TaskSpec, task: Task) -> tuple:
    io_format = spec.io_format

    return (
        task,
        tuple(spec.label_set or []),
        tuple(spec.constraints or []),
        tuple(io_format.input_constraints or []),
        tuple(io_format.output_constraints or []),
        io_format.output_description or "",
    )


class FormatValidator:
    def __init__(self) -> None:
        self._model_cache: dict[tuple, type[ExampleBase]] = {}

    def validate(
            self,
            raw_examples: list[Any],
            spec: TaskSpec,
            task: Task,
    ) -> tuple[list[ExampleBase], list[Any]]:
        model = self._get_or_build_model(spec, task)

        valid: list[ExampleBase] = []
        invalid: list[Any] = []

        for raw in raw_examples:
            try:
                data = self._to_validation_data(raw)
                valid.append(model.model_validate(data))

            except (
                    ValidationError,
                    AttributeError,
                    TypeError,
                    ValueError,
            ) as exc:
                logger.info(
                    "Rejected example (structural): %s | error=%s",
                    raw,
                    exc,
                )
                invalid.append(raw)

        return valid, invalid

    @staticmethod
    def _to_validation_data(raw: Any) -> dict[str, Any]:
        if isinstance(raw, BaseModel):
            return raw.model_dump()

        if isinstance(raw, dict):
            return raw

        return {
            "input": getattr(raw, "input"),
            "output": getattr(raw, "output"),
        }

    def _get_or_build_model(
            self,
            spec: TaskSpec,
            task: Task,
    ) -> type[ExampleBase]:
        key = _model_cache_key(spec, task)
        model = self._model_cache.get(key)

        if model is None:
            model = build_example_model(spec, task)
            self._model_cache[key] = model

        return model


class Deduplicator:
    def __init__(
            self,
            near_dup_threshold: float = 0.85,
            shingle_size: int = 3,
            enable_near_dup: bool = True,
    ) -> None:
        if not 0.0 <= near_dup_threshold <= 1.0:
            raise ValueError(
                "near_dup_threshold must be between 0.0 and 1.0"
            )

        if shingle_size < 1:
            raise ValueError("shingle_size must be at least 1")

        self._seen_inputs: set[str] = set()
        self._accepted_shingles: list[set[str]] = []
        self._shingle_index: dict[str, set[int]] = {}

        self._near_dup_threshold = near_dup_threshold
        self._shingle_size = shingle_size
        self._enable_near_dup = enable_near_dup

    def dedupe_exact_pairs_within_batch(self, examples: list[ExampleBase]) -> list[ExampleBase]:
        seen_pairs: set[tuple[str, str]] = set()
        fresh: list[ExampleBase] = []

        for example in examples:
            pair_key = (
                _normalize_text(example.input),
                _normalize_text(example.output),
            )

            if pair_key in seen_pairs:
                logger.info(
                    "Rejected example "
                    "(exact input/output duplicate within batch): %s",
                    example,
                )
                continue

            seen_pairs.add(pair_key)
            fresh.append(example)

        return fresh

    def filter(
            self,
            examples: list[ExampleBase],
            *,
            limit: int | None = None,
    ) -> list[ExampleBase]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        if limit == 0:
            return []

        accepted: list[ExampleBase] = []

        for example in examples:
            if limit is not None and len(accepted) >= limit:
                break

            normalized_input, shingles = self._prepare_input(example)

            if normalized_input in self._seen_inputs:
                logger.info("Rejected example (exact input duplicate): %s", example.input)
                continue

            match_score = self._best_candidate_score(shingles)

            if (
                    self._enable_near_dup
                    and shingles
                    and match_score >= self._near_dup_threshold
            ):
                logger.info("Rejected example "
                            "(near input duplicate, jaccard=%.2f): %s", match_score, example.input)
                continue

            self._accept(normalized_input, shingles)
            accepted.append(example)

        return accepted

    def reset(self) -> None:
        self._seen_inputs.clear()
        self._accepted_shingles.clear()
        self._shingle_index.clear()

    def _prepare_input(
            self,
            example: ExampleBase,
    ) -> tuple[str, set[str]]:
        normalized_input = _normalize_text(example.input)

        if not self._enable_near_dup:
            return normalized_input, set()

        return (
            normalized_input,
            _shingles(normalized_input, self._shingle_size),
        )

    def _best_candidate_score(
            self,
            shingles: set[str],
    ) -> float:
        if not self._enable_near_dup or not shingles:
            return 0.0

        candidate_indices: set[int] = set()

        for shingle in shingles:
            candidate_indices.update(
                self._shingle_index.get(shingle, set())
            )

        if not candidate_indices:
            return 0.0

        return max(_jaccard(shingles, self._accepted_shingles[index])
                   for index in candidate_indices)

    def _accept(
            self,
            normalized_input: str,
            shingles: set[str],
    ) -> None:
        self._seen_inputs.add(normalized_input)

        if not self._enable_near_dup:
            return

        new_index = len(self._accepted_shingles)
        self._accepted_shingles.append(shingles)

        for shingle in shingles:
            self._shingle_index.setdefault(
                shingle,
                set(),
            ).add(new_index)
