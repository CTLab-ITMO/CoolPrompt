"""Prompt embeddings, evaluate-first (mu + lambda) retention and k-DPP ensembles."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence

import numpy as np

from coolprompt.utils.logging_config import logger

from .population import Candidate


class PromptEmbedder:
    """Sentence-BERT embeddings with a dependency-free fallback.

    The paper measures near-duplicates with Sentence-BERT cosine similarity.
    When ``sentence-transformers`` or the model is unavailable, character
    trigram TF vectors keep retention and ensembling working.
    """

    def __init__(self, model_name: str | None) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = model_name is None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - depends on environment
                logger.warning(
                    "RIDER: Sentence-BERT unavailable (%s); using character n-grams",
                    exc,
                )
                self._failed = True
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            model = self._load()
            if model is not None:
                vectors = model.encode(
                    list(missing), normalize_embeddings=True, show_progress_bar=False
                )
                for text, vector in zip(missing, vectors):
                    self._cache[text] = np.asarray(vector, dtype=float)
            else:
                for text in missing:
                    self._cache[text] = _ngram_vector(text)
        if self._model is None:
            return _stack_sparse([self._cache[t] for t in texts])
        return np.stack([self._cache[t] for t in texts])

    def similarity(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.embed(texts)
        sim = vectors @ vectors.T
        return np.clip(sim, -1.0, 1.0)


def _ngram_vector(text: str, n: int = 3) -> Counter:
    text = " ".join(text.lower().split())
    return Counter(text[i : i + n] for i in range(max(len(text) - n + 1, 1)))


def _stack_sparse(counters: list) -> np.ndarray:
    vocab = {key: i for i, key in enumerate({k for c in counters for k in c})}
    matrix = np.zeros((len(counters), max(len(vocab), 1)))
    for row, counter in enumerate(counters):
        for key, value in counter.items():
            matrix[row, vocab[key]] = value
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _within_tolerance(a: float, b: float, tolerance: float) -> bool:
    """Fitness values within ``tolerance`` (relative, 1% in the paper)."""

    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) <= tolerance * scale


def mu_plus_lambda_select(
    pool: Sequence[Candidate],
    size: int,
    similarity: Callable[[Sequence[str]], np.ndarray],
    *,
    threshold: float,
    tolerance: float,
    elite: int = 0,
) -> list[Candidate]:
    """Fitness-first retention with diversity as a tiebreaker (Sec. III-E).

    Parents and evaluated offspring are pooled and sorted by fitness. The top
    ``elite`` distinct prompts are always kept. After that, a candidate whose
    cosine similarity to an already kept prompt exceeds ``threshold`` is
    skipped as a near-duplicate only when its fitness is within ``tolerance``
    of that prompt. Because the pool is visited best-first, a kept prompt is
    never fitter than a later similar one, so diversity never discards a
    genuinely better prompt; skipped near-duplicates fill any remaining slots.
    """

    unique: dict[str, Candidate] = {}
    for cand in pool:
        kept = unique.get(cand.text)
        if kept is None or cand.fitness > kept.fitness:
            unique[cand.text] = cand
    ranked = sorted(unique.values(), key=lambda c: c.fitness, reverse=True)
    if not ranked:
        return []
    sim = similarity([c.text for c in ranked])
    index = {id(c): i for i, c in enumerate(ranked)}
    selected: list[Candidate] = []
    skipped: list[Candidate] = []
    for cand in ranked:
        if len(selected) >= size:
            break
        i = index[id(cand)]
        if len(selected) < elite:
            selected.append(cand)
            continue
        twin = next(
            (kept for kept in selected if sim[i, index[id(kept)]] > threshold),
            None,
        )
        if twin is not None and _within_tolerance(
            cand.fitness, twin.fitness, tolerance
        ):
            skipped.append(cand)
        else:
            selected.append(cand)
    # Too few distinct prompts: fill with the best near-duplicates.
    for cand in skipped:
        if len(selected) >= size:
            break
        if cand not in selected:
            selected.append(cand)
    return sorted(selected, key=lambda c: c.fitness, reverse=True)


def drop_near_duplicates(
    candidates: Sequence[Candidate],
    similarity: Callable[[Sequence[str]], np.ndarray],
    threshold: float,
) -> list[Candidate]:
    """Keep the first of every group of prompts more similar than ``threshold``."""

    items = list(candidates)
    if len(items) < 2:
        return items
    sim = similarity([c.text for c in items])
    kept: list[int] = []
    for i in range(len(items)):
        if all(sim[i, j] <= threshold for j in kept):
            kept.append(i)
    return [items[i] for i in kept]


def greedy_kdpp(
    candidates: Sequence[Candidate],
    k: int,
    similarity: Callable[[Sequence[str]], np.ndarray],
    quality_weight: float = 1.0,
) -> list[Candidate]:
    """Greedy MAP approximation of a k-DPP with L_ij = q_i S_ij q_j (Sec. III-H)."""

    items = list(candidates)
    if len(items) <= k:
        return sorted(items, key=lambda c: c.fitness, reverse=True)
    sim = similarity([c.text for c in items])
    fitness = np.array([c.fitness for c in items], dtype=float)
    span = fitness.max() - fitness.min()
    normalized = (fitness - fitness.min()) / span if span > 0 else np.ones_like(fitness)
    quality = np.exp(quality_weight * normalized)
    kernel = quality[:, None] * sim * quality[None, :] + 1e-6 * np.eye(len(items))
    chosen: list[int] = [int(np.argmax(fitness))]
    while len(chosen) < k:
        best_gain, best_index = -math.inf, None
        for j in range(len(items)):
            if j in chosen:
                continue
            idx = chosen + [j]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(idx, idx)])
            gain = logdet if sign > 0 else -math.inf
            if gain > best_gain:
                best_gain, best_index = gain, j
        if best_index is None:
            break
        chosen.append(best_index)
    return [items[i] for i in chosen]
