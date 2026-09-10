"""Structural validation, deduplication, and novelty filtering."""

from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from html import unescape
from typing import Any

from pydantic import BaseModel, ValidationError
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coolprompt.spec_generator.models import Example, TaskSpec
from coolprompt.utils.logging_config import logger

_WORD_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary text for comparison."""

    text = unescape(str(value))
    text = unicodedata.normalize("NFKC", text).casefold()

    return " ".join(text.split())


def _normalize_output(value: Any) -> str:
    """Normalize output values, including numeric outputs."""

    text = str(value).strip()

    try:
        number = Decimal(text)
    except InvalidOperation:
        return _normalize_text(text)

    if not number.is_finite():
        return _normalize_text(text)

    return (
        str(number.to_integral())
        if number == number.to_integral()
        else format(number.normalize(), "f")
    )


def _tokens(text: str) -> list[str]:
    """Tokenize text for lightweight structural comparison."""

    normalized = unicodedata.normalize("NFKC", unescape(text))

    return [token.casefold() for token in _WORD_RE.findall(normalized)]


def _canonical_concept_set(value: str) -> tuple[str, ...] | None:
    """Return a canonical representation of list-like concept inputs."""

    try:
        parsed = ast.literal_eval(unescape(value).strip())
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, (list, tuple)):
        return None

    normalized = sorted(
        text for item in parsed if (text := str(item).strip().casefold())
    )

    return tuple(normalized) or None


def _structural_signature(example: Example) -> str | None:
    """Return output structure with concepts and numbers masked."""

    output_tokens = _tokens(example.output)

    if len(output_tokens) < 6:
        return None

    input_tokens = {token for token in _tokens(example.input) if len(token) >= 2}

    signature = [
        (
            "__concept__"
            if token in input_tokens
            else "__number__" if _NUMBER_RE.match(token) else token
        )
        for token in output_tokens
    ]

    return " ".join(signature)


class ExampleValidator:
    """Validate generated examples against a task specification."""

    def validate(
        self, raw_examples: list[Any], spec: TaskSpec
    ) -> tuple[list[Example], list[Any]]:
        """Split raw candidates into valid and invalid examples."""

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
        """Normalize a classification label."""

        if not spec.labels:
            return example

        labels = {label.casefold(): label for label in spec.labels}
        canonical = labels.get(example.output.casefold())

        if canonical is None:
            raise ValueError(
                f"Output {example.output!r} is not in label set {spec.labels!r}."
            )

        return (
            example
            if canonical == example.output
            else Example(input=example.input, output=canonical)
        )

    @staticmethod
    def _to_dict(raw: Any) -> dict[str, Any]:
        """Preserve only public example fields."""

        payload = (
            raw.model_dump()
            if isinstance(raw, BaseModel)
            else (
                raw
                if isinstance(raw, dict)
                else {
                    "input": getattr(raw, "input", None),
                    "output": getattr(raw, "output", None),
                }
            )
        )

        input_value = payload.get("input")

        return {
            "input": (
                unescape(input_value) if isinstance(input_value, str) else input_value
            ),
            "output": payload.get("output"),
        }


class Deduplicator:
    """Remove exact, near, semantic, structural, and concept-set duplicates."""

    def __init__(
        self,
        near_dup_threshold: float = 0.80,
        enable_near_dup: bool = True,
        *,
        enable_semantic_novelty: bool = False,
        semantic_threshold: float = 0.72,
        enable_structural_novelty: bool = False,
        structural_threshold: float = 0.78,
    ) -> None:
        """Configure duplicate and novelty thresholds and vectorizers."""

        self._seen_inputs: set[str] = set()
        self._seen_concept_sets: set[tuple[str, ...]] = set()
        self._char_matrix: csr_matrix | None = None
        self._semantic_matrix: csr_matrix | None = None
        self._structure_matrix: csr_matrix | None = None

        thresholds = {
            "near_dup_threshold": near_dup_threshold,
            "semantic_threshold": semantic_threshold,
            "structural_threshold": structural_threshold,
        }

        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        self._near_dup_threshold = near_dup_threshold
        self._enable_near_dup = enable_near_dup
        self._enable_semantic_novelty = enable_semantic_novelty
        self._semantic_threshold = semantic_threshold
        self._enable_structural_novelty = enable_structural_novelty
        self._structural_threshold = structural_threshold

        self._char_vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2**18,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )

        self._semantic_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=2**18,
            lowercase=True,
            alternate_sign=False,
            norm="l2",
        )

        self._structure_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            n_features=2**16,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
            token_pattern=(r"(?u)\b\w[\w_'-]*\b"),
        )

        self.reset()

    @staticmethod
    def dedupe_exact_pairs_within_batch(examples: list[Example]) -> list[Example]:
        """Remove exact input/output duplicates within one batch."""

        seen: set[tuple[str, str]] = set()
        unique: list[Example] = []

        for example in examples:
            key = (_normalize_text(example.input), _normalize_output(example.output))

            if key in seen:
                continue

            seen.add(key)
            unique.append(example)

        return unique

    def filter(
        self, examples: list[Example], *, limit: int | None = None
    ) -> list[Example]:
        """Filter candidates against examples already accepted by this instance."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        accepted: list[Example] = []

        for example in examples:
            if limit is not None and len(accepted) >= limit:
                break

            normalized_input = _normalize_text(example.input)
            concept_set = _canonical_concept_set(example.input)

            if concept_set in self._seen_concept_sets:
                logger.info("Rejected duplicate concept set: %s", example.input)
                continue

            if normalized_input in self._seen_inputs:
                logger.info("Rejected duplicate input: %s", example.input)
                continue

            char_vector = (
                self._char_vectorizer.transform([normalized_input])
                if normalized_input
                else None
            )

            semantic_text = _normalize_text(f"{example.input} {example.output}")

            semantic_vector = (
                self._semantic_vectorizer.transform([semantic_text])
                if self._enable_semantic_novelty and semantic_text
                else None
            )

            structure = _structural_signature(example)
            structure_vector = (
                self._structure_vectorizer.transform([structure])
                if self._enable_structural_novelty and structure
                else None
            )

            checks = (
                (
                    self._enable_near_dup,
                    char_vector,
                    self._char_matrix,
                    self._near_dup_threshold,
                    "near-duplicate",
                ),
                (
                    self._enable_semantic_novelty,
                    semantic_vector,
                    self._semantic_matrix,
                    self._semantic_threshold,
                    "semantic repetition",
                ),
                (
                    self._enable_structural_novelty,
                    structure_vector,
                    self._structure_matrix,
                    self._structural_threshold,
                    "structural repetition",
                ),
            )

            rejected = False

            for enabled, vector, matrix, threshold, reason in checks:
                if enabled and self._best_similarity(vector, matrix) >= threshold:
                    logger.info("Rejected %s: %s", reason, example.input)
                    rejected = True
                    break

            if rejected:
                continue

            self._seen_inputs.add(normalized_input)

            if concept_set is not None:
                self._seen_concept_sets.add(concept_set)

            self._char_matrix = self._append(self._char_matrix, char_vector)
            self._semantic_matrix = self._append(self._semantic_matrix, semantic_vector)
            self._structure_matrix = self._append(
                self._structure_matrix, structure_vector
            )

            accepted.append(example)

        return accepted

    @staticmethod
    def _append(
        matrix: csr_matrix | None,
        vector: csr_matrix | None,
    ) -> csr_matrix | None:
        """Append a sparse vector to the comparison matrix."""

        if vector is None:
            return matrix

        return vector if matrix is None else vstack((matrix, vector))

    @staticmethod
    def _best_similarity(
        vector: csr_matrix | None,
        matrix: csr_matrix | None,
    ) -> float:
        """Return maximum cosine similarity against previously accepted vectors."""

        if vector is None or matrix is None:
            return 0.0

        similarities = cosine_similarity(vector, matrix)[0]
        return float(similarities.max()) if similarities.size else 0.0

    def reset(self) -> None:
        """Reset deduplication history."""

        self._seen_inputs.clear()
        self._seen_concept_sets.clear()
        self._char_matrix = None
        self._semantic_matrix = None
        self._structure_matrix = None
