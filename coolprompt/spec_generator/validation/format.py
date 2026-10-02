"""Example validation and deduplication."""

from __future__ import annotations

import unicodedata
from html import unescape
from typing import Any

from pydantic import BaseModel, ValidationError
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coolprompt.spec_generator.models import Example, TaskSpec
from coolprompt.utils.logging_config import logger


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary text for comparison."""

    text = unescape(str(value))
    text = unicodedata.normalize("NFKC", text).casefold()

    return " ".join(text.split())


class ExampleValidator:
    """Validate generated examples against a task specification."""

    def validate(
        self,
        raw_examples: list[Any],
        spec: TaskSpec,
    ) -> tuple[list[Example], list[Any]]:
        """Split raw candidates into valid and invalid examples."""

        valid: list[Example] = []
        invalid: list[Any] = []

        for raw in raw_examples:
            try:
                example = Example.model_validate(self._to_dict(raw))

                if not example.input.strip():
                    raise ValueError("Empty input is not allowed.")

                if not example.output.strip() and not spec.allow_empty_output:
                    raise ValueError("Empty output is not allowed.")

                valid.append(self._normalize_label(example, spec))

            except (ValidationError, AttributeError, TypeError, ValueError) as exc:
                logger.info("Rejected example: %s | error=%s", raw, exc)
                invalid.append(raw)

        return valid, invalid

    @staticmethod
    def _normalize_label(example: Example, spec: TaskSpec) -> Example:
        """Normalize a classification label."""

        if not spec.labels:
            return example

        labels = {label.casefold(): label for label in spec.labels}

        canonical = labels.get(example.output.strip().casefold())

        if canonical is None:
            raise ValueError(
                f"Output {example.output!r} is not in label set {spec.labels!r}."
            )

        if canonical == example.output:
            return example

        return Example(
            input=example.input,
            output=canonical,
        )

    @staticmethod
    def _to_dict(raw: Any) -> dict[str, Any]:
        """Preserve only public example fields."""

        if isinstance(raw, BaseModel):
            payload = raw.model_dump()
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = {
                "input": getattr(raw, "input", None),
                "output": getattr(raw, "output", None),
            }

        input_value = payload.get("input")

        return {
            "input": (
                unescape(input_value) if isinstance(input_value, str) else input_value
            ),
            "output": payload.get("output"),
        }


class Deduplicator:
    """Filter exact and near-duplicate inputs."""

    def __init__(
        self,
        char_threshold: float = 0.80,
        word_threshold: float = 0.75,
    ) -> None:
        """Configure near-duplicate filtering."""

        if not 0.0 <= char_threshold <= 1.0:
            raise ValueError("char_threshold must be between 0 and 1")

        if not 0.0 <= word_threshold <= 1.0:
            raise ValueError("word_threshold must be between 0 and 1")

        self._char_threshold = char_threshold
        self._word_threshold = word_threshold

        self._seen_inputs: set[str] = set()

        self._char_matrix: csr_matrix | None = None
        self._word_matrix: csr_matrix | None = None

        self._char_vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2**18,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )

        self._word_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=2**18,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )

    def _is_near_duplicate(
        self,
        char_vector: csr_matrix | None,
        word_vector: csr_matrix | None,
    ) -> bool:
        """Return whether a candidate is near-duplicate of an accepted input."""

        if (
            char_vector is None
            or word_vector is None
            or self._char_matrix is None
            or self._word_matrix is None
        ):
            return False

        char_similarities = cosine_similarity(
            char_vector,
            self._char_matrix,
        )[0]

        word_similarities = cosine_similarity(
            word_vector,
            self._word_matrix,
        )[0]

        return bool(
            (
                (char_similarities >= self._char_threshold)
                & (word_similarities >= self._word_threshold)
            ).any()
        )

    def accept(self, example: Example) -> bool:
        """Accept an example unless its input duplicates an accepted input."""

        normalized_input = _normalize_text(example.input)

        if normalized_input in self._seen_inputs:
            logger.info("Rejected duplicate input: %s", example.input)
            return False

        char_vector = (
            self._char_vectorizer.transform([normalized_input])
            if normalized_input
            else None
        )

        word_vector = (
            self._word_vectorizer.transform([normalized_input])
            if normalized_input
            else None
        )

        if self._is_near_duplicate(char_vector, word_vector):
            logger.info("Rejected near-duplicate: %s", example.input)
            return False

        self._seen_inputs.add(normalized_input)

        self._char_matrix = self._append(
            self._char_matrix,
            char_vector,
        )
        self._word_matrix = self._append(
            self._word_matrix,
            word_vector,
        )

        return True

    @staticmethod
    def _append(
        matrix: csr_matrix | None,
        vector: csr_matrix | None,
    ) -> csr_matrix | None:
        """Append a sparse vector to the comparison matrix."""

        if vector is None:
            return matrix

        return vector if matrix is None else vstack((matrix, vector))

    def reset(self) -> None:
        """Reset deduplication history."""

        self._seen_inputs.clear()
        self._char_matrix = None
        self._word_matrix = None
