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

        if not number.is_finite():
            return _normalize_text(text)

        if number == number.to_integral():
            return str(number.to_integral())

        return format(
            number.normalize(),
            "f",
        )

    except InvalidOperation:
        return _normalize_text(text)


def _tokens(text: str) -> list[str]:
    """Tokenize text for lightweight structural comparison."""

    normalized = unicodedata.normalize(
        "NFKC",
        unescape(text),
    )

    return [
        token.casefold()
        for token in _WORD_RE.findall(normalized)
    ]


def _canonical_concept_set(
    value: str,
) -> tuple[str, ...] | None:
    """Return a canonical representation of list-like concept inputs.

    Examples:

    ['innovation', 'technology', 'future', 'drive']

    and

    ['future', 'drive', 'technology', 'innovation']

    both become the same canonical tuple.

    Non-list-like inputs return None so this mechanism remains harmless
    for tasks that do not use concept lists.
    """

    try:
        parsed = ast.literal_eval(unescape(value).strip())

    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, (list, tuple)):
        return None

    normalized = [
        str(item).strip().casefold()
        for item in parsed
        if str(item).strip()
    ]

    if not normalized:
        return None

    return tuple(sorted(normalized))


def _structural_signature(
    example: Example,
) -> str | None:
    """Approximate output structure while masking input concepts.

    Useful for sentence-generation tasks such as CommonGen.
    Short outputs, labels, and simple numeric answers effectively
    disable structural comparison.
    """

    output_tokens = _tokens(example.output)

    if len(output_tokens) < 6:
        return None

    input_tokens = {
        token
        for token in _tokens(example.input)
        if len(token) >= 2
    }

    signature: list[str] = []

    for token in output_tokens:
        if token in input_tokens:
            signature.append("__concept__")

        elif _NUMBER_RE.match(token):
            signature.append("__number__")

        else:
            signature.append(token)

    return " ".join(signature)


class ExampleValidator:
    """Validate generated examples against a task specification."""

    def __init__(
        self,
        *,
        min_references: int = 0,
    ) -> None:
        if min_references < 0:
            raise ValueError("min_references must be non-negative")

        self._min_references = min_references

    def validate(
        self,
        raw_examples: list[Any],
        spec: TaskSpec,
    ) -> tuple[list[Example], list[Any]]:
        """Validate generated examples and split valid/invalid candidates."""

        valid: list[Example] = []
        invalid: list[Any] = []

        for raw in raw_examples:
            try:
                example = Example.model_validate(self._to_dict(raw))

                if len(example.references) < self._min_references:
                    raise ValueError(
                        "Expected at least "
                        f"{self._min_references} "
                        "alternative references, "
                        f"received {len(example.references)}."
                    )

                example = self._normalize_label(example, spec)

                valid.append(example)

            except (ValidationError, AttributeError, TypeError, ValueError) as exc:
                logger.info("Rejected example: %s | error=%s", raw, exc)
                invalid.append(raw)

        return valid, invalid

    @staticmethod
    def _normalize_label(
        example: Example,
        spec: TaskSpec,
    ) -> Example:
        """Normalize classification labels while preserving references."""

        if not spec.labels:
            return example

        labels = {
            label.casefold(): label
            for label in spec.labels
        }

        canonical = labels.get(example.output.casefold())

        if canonical is None:
            raise ValueError(
                f"Output {example.output!r} "
                f"is not in label set {spec.labels!r}."
            )

        if canonical == example.output:
            return example

        return Example(
            input=example.input,
            output=canonical,
            references=example.references,
        )

    @staticmethod
    def _to_dict(
        raw: Any,
    ) -> dict[str, Any]:
        """Preserve public example fields while dropping generation metadata."""

        if isinstance(raw, BaseModel):
            payload = raw.model_dump()

        elif isinstance(raw, dict):
            payload = raw

        else:
            payload = {"input": getattr(raw, "input"),
                "output": getattr(raw, "output"),
                "references": getattr(raw, "references", ()),
            }

        input_value = payload.get("input")

        if isinstance(input_value, str):
            input_value = unescape(input_value)

        return {
            "input": input_value,
            "output": payload.get("output"), "references": payload.get("references") or ()}


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
        thresholds = {
            "near_dup_threshold": near_dup_threshold,
            "semantic_threshold": semantic_threshold,
            "structural_threshold": structural_threshold,
        }

        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

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
            token_pattern=(
                r"(?u)\b\w[\w_'-]*\b"
            ),
        )

        self.reset()

    @staticmethod
    def dedupe_exact_pairs_within_batch(
        examples: list[Example],
    ) -> list[Example]:
        """Remove exact input/output duplicate pairs within one model response."""

        seen: set[tuple[str, str]] = set()
        result: list[Example] = []

        for example in examples:
            key = (_normalize_text(example.input),
                _normalize_output(example.output))

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
        """Filter candidates against examples already accepted by this instance."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        accepted: list[Example] = []

        for example in examples:
            if limit is not None and len(accepted) >= limit:
                break

            normalized_input = _normalize_text(example.input)
            concept_set = _canonical_concept_set(example.input)

            if (concept_set is not None
                and concept_set
                in self._seen_concept_sets):
                logger.info("Rejected duplicate concept set: %s", example.input)
                continue

            char_vector = (self._char_vectorizer.transform([normalized_input])
                if normalized_input else None)

            semantic_text = _normalize_text(
                f"{example.input} "
                f"{example.output}"
            )

            semantic_vector = (self._semantic_vectorizer.transform([semantic_text])
                if (self._enable_semantic_novelty and semantic_text)
                else None)

            structure = _structural_signature(example)

            structure_vector = (self._structure_vectorizer.transform([structure])
                if (self._enable_structural_novelty and structure) else None)

            if normalized_input in self._seen_inputs:
                logger.info("Rejected duplicate input: %s", example.input)
                continue

            if (self._enable_near_dup and self._best_similarity(char_vector, self._char_matrix)
                >= self._near_dup_threshold):
                logger.info("Rejected near-duplicate input: %s", example.input)
                continue

            if (self._enable_semantic_novelty and self._best_similarity(semantic_vector, self._semantic_matrix)
                >= self._semantic_threshold):
                logger.info("Rejected semantic repetition: %s", example.input)
                continue

            if (self._enable_structural_novelty and self._best_similarity(structure_vector, self._structure_matrix)
                >= self._structural_threshold):
                logger.info("Rejected structural repetition: %s", example.input)
                continue

            self._seen_inputs.add(normalized_input)

            if concept_set is not None:
                self._seen_concept_sets.add(concept_set)

            self._char_matrix = self._append(self._char_matrix, char_vector)
            self._semantic_matrix = self._append(self._semantic_matrix, semantic_vector)
            self._structure_matrix = self._append(self._structure_matrix, structure_vector)

            accepted.append(example)

        return accepted

    @staticmethod
    def _append(
        matrix: csr_matrix | None,
        vector: csr_matrix | None,
    ) -> csr_matrix | None:
        """Append one sparse vector to a stored comparison matrix."""

        if vector is None:
            return matrix

        if matrix is None:
            return vector

        return vstack([matrix, vector])

    @staticmethod
    def _best_similarity(
        vector: csr_matrix | None,
        matrix: csr_matrix | None,
    ) -> float:
        """Return maximum cosine similarity against previously accepted vectors."""

        if vector is None or matrix is None:
            return 0.0

        similarities = cosine_similarity(vector, matrix)[0]

        if not similarities.size:
            return 0.0

        return float(similarities.max())

    def reset(self) -> None:
        """Reset all deduplication history."""

        self._seen_inputs: set[str] = set()
        self._seen_concept_sets: set[tuple[str, ...]] = set()
        self._char_matrix: (csr_matrix | None) = None
        self._semantic_matrix: (csr_matrix | None) = None
        self._structure_matrix: (csr_matrix | None) = None