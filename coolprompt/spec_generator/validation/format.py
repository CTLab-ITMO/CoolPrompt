"""Structural validation and deduplication for generated examples."""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ValidationError
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coolprompt.spec_generator.models import Example, TaskSpec
from coolprompt.utils.logging_config import logger


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _normalize_output(value: Any) -> str:
    text = str(value).strip()
    try:
        number = Decimal(text)
        if not number.is_finite():
            return _normalize_text(text)
        if number == number.to_integral():
            return str(number.to_integral())
        return format(number.normalize(), "f")
    except InvalidOperation:
        return _normalize_text(text)


class ExampleValidator:
    """Validate generated examples against a task specification."""

    def validate(
        self,
        raw_examples: list[Any],
        spec: TaskSpec,
    ) -> tuple[list[Example], list[Any]]:
        valid: list[Example] = []
        invalid: list[Any] = []

        for raw in raw_examples:
            try:
                example = Example.model_validate(self._to_dict(raw))
                valid.append(self._normalize_label(example, spec))
            except (ValidationError, AttributeError, TypeError, ValueError) as exc:
                logger.info("Rejected example: %s | error=%s", raw, exc)
                invalid.append(raw)

        return valid, invalid

    @staticmethod
    def _normalize_label(example: Example, spec: TaskSpec) -> Example:
        if not spec.labels:
            return example

        labels = {label.casefold(): label for label in spec.labels}
        canonical = labels.get(example.output.casefold())
        if canonical is None:
            raise ValueError(f"Output {example.output!r} is not in label set {spec.labels!r}.")

        if canonical == example.output:
            return example
        return Example(input=example.input, output=canonical)

    @staticmethod
    def _to_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, BaseModel):
            return raw.model_dump()
        if isinstance(raw, dict):
            return raw
        return {
            "input": getattr(raw, "input"),
            "output": getattr(raw, "output"),
        }


class Deduplicator:
    """Remove exact and near-duplicate inputs across validation rounds."""

    def __init__(
        self,
        near_dup_threshold: float = 0.8,
        enable_near_dup: bool = True,
    ) -> None:
        if not 0.0 <= near_dup_threshold <= 1.0:
            raise ValueError("near_dup_threshold must be between 0 and 1")

        self._threshold = near_dup_threshold
        self._enable_near_dup = enable_near_dup
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2**18,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )
        self.reset()

    @staticmethod
    def dedupe_exact_pairs_within_batch(examples: list[Example]) -> list[Example]:
        seen: set[tuple[str, str]] = set()
        result: list[Example] = []

        for example in examples:
            key = (_normalize_text(example.input), _normalize_output(example.output))
            if key in seen:
                continue
            seen.add(key)
            result.append(example)

        return result

    def filter(
        self,
        examples: list[Example],
        *,
        limit: int | None = None,
    ) -> list[Example]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        accepted: list[Example] = []
        for example in examples:
            if limit is not None and len(accepted) >= limit:
                break

            normalized = _normalize_text(example.input)
            vector = self._vectorize(normalized)
            if normalized in self._seen_inputs:
                continue
            if self._best_similarity(vector) >= self._threshold:
                continue

            self._seen_inputs.add(normalized)
            if vector is not None:
                self._matrix = vector if self._matrix is None else vstack([self._matrix, vector])
            accepted.append(example)

        return accepted

    def _vectorize(self, text: str) -> csr_matrix | None:
        if not self._enable_near_dup or not text:
            return None
        return self._vectorizer.transform([text])

    def _best_similarity(self, vector: csr_matrix | None) -> float:
        if vector is None or self._matrix is None:
            return 0.0
        similarities = cosine_similarity(vector, self._matrix)[0]
        return float(similarities.max()) if similarities.size else 0.0

    def reset(self) -> None:
        self._seen_inputs: set[str] = set()
        self._matrix: csr_matrix | None = None
