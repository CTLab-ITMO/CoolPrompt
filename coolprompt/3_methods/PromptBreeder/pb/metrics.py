"""Evaluation metrics for PromptBreeder fitness scoring.

Three metrics are exposed:

* ``bert_score`` — F1 of BERTScore between model output and reference.
* ``exact_match`` — normalized exact-match (1.0 / 0.0).
* ``f1_mera`` — SQuAD-style token-level F1.

Each scorer takes ``(prediction, reference, example=None)`` and returns a
float in ``[0, 1]``. The optional ``example`` argument lets dataset-specific
logic (e.g. GSM8K numeric extraction) refine the comparison.
"""

from __future__ import annotations

import logging
import re
import string
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


SUPPORTED_METRICS: List[str] = ["bert_score", "exact_match", "f1_mera"]


# gsm8k → exact_match, tweeteval → f1_mera, everything else → bert_score
DEFAULT_METRIC_BY_DATASET: Dict[str, str] = {
    "gsm8k": "exact_match",
    "tweeteval": "f1_mera",
    "squad_v2": "bert_score",
    "common_gen": "bert_score",
    "xsum": "bert_score",
    "mediqa": "bert_score",
    "code_to_text": "bert_score",
    "concode": "bert_score",
}


# --------------------------------------------------------------------- helpers
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def _extract_last_number(text: str) -> Optional[str]:
    cleaned = (text or "").replace(",", "")
    matches = _NUMBER_RE.findall(cleaned)
    return matches[-1] if matches else None


# ----------------------------------------------------------------- exact_match
def exact_match(prediction: str, reference: str, example: Optional[Dict] = None) -> float:
    """Return 1.0 if prediction matches reference, else 0.0.

    For numeric references (e.g. GSM8K targets), comparison is done on the
    last number extracted from each string. Otherwise, both strings are
    normalized (lowercase, punctuation stripped, whitespace collapsed).
    """
    ref_num = _extract_last_number(reference)
    if ref_num is not None and _NUMBER_RE.fullmatch(ref_num):
        # Treat references that boil down to a single number numerically.
        ref_only_number = _normalize(reference).replace(" ", "") == ref_num.replace("-", "")
        if ref_only_number or (example and (example.get("dataset") == "gsm8k")):
            pred_num = _extract_last_number(prediction)
            if pred_num is None:
                return 0.0
            try:
                return 1.0 if float(pred_num) == float(ref_num) else 0.0
            except (ValueError, TypeError):
                return 0.0
        # Fall through to string compare otherwise.
    pred_norm = _normalize(prediction)
    ref_norm = _normalize(reference)
    if not ref_norm:
        return 1.0 if not pred_norm else 0.0
    return 1.0 if pred_norm == ref_norm else 0.0


# --------------------------------------------------------------------- f1_mera
def f1_mera(prediction: str, reference: str, example: Optional[Dict] = None) -> float:
    """SQuAD-style token-level F1 over normalized tokens."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    # Multiset intersection.
    common: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in ref_tokens:
        if pred_counts.get(tok, 0) > 0:
            common[tok] = common.get(tok, 0) + 1
            pred_counts[tok] -= 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------ bert_score
_BERTSCORE_LOCK = threading.Lock()
_BERTSCORE_MODEL = None  # type: ignore
_BERTSCORE_FAILED = False


def _get_bertscore_fn():
    """Lazy-load and cache the bert_score.score function (no-op if unavailable)."""
    global _BERTSCORE_MODEL, _BERTSCORE_FAILED
    if _BERTSCORE_FAILED:
        return None
    if _BERTSCORE_MODEL is not None:
        return _BERTSCORE_MODEL
    with _BERTSCORE_LOCK:
        if _BERTSCORE_MODEL is not None:
            return _BERTSCORE_MODEL
        try:
            from bert_score import score as _score  # type: ignore
            _BERTSCORE_MODEL = _score
            return _BERTSCORE_MODEL
        except Exception as exc:  # pragma: no cover - import error path
            logger.warning(
                "bert_score is unavailable (%s); falling back to token F1.", exc
            )
            _BERTSCORE_FAILED = True
            return None


def bert_score(prediction: str, reference: str, example: Optional[Dict] = None) -> float:
    """Compute the BERTScore F1 between prediction and reference.

    Falls back to :func:`f1_mera` if the ``bert-score`` package or its model
    weights are unavailable at runtime (e.g. offline environments).
    """
    pred = (prediction or "").strip()
    ref = (reference or "").strip()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    score_fn = _get_bertscore_fn()
    if score_fn is None:
        return f1_mera(prediction, reference, example)

    try:
        import os
        model_type = os.getenv("PB_BERTSCORE_MODEL", "distilbert-base-uncased")
        _, _, f1 = score_fn(
            [pred],
            [ref],
            model_type=model_type,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        value = float(f1[0].item())
        # BERTScore can be slightly negative; clip to [0, 1].
        return max(0.0, min(1.0, value))
    except Exception as exc:  # pragma: no cover - runtime fallback
        logger.warning("bert_score failed (%s); falling back to token F1.", exc)
        return f1_mera(prediction, reference, example)


# ------------------------------------------------------------------- registry
_SCORERS: Dict[str, Callable[[str, str, Optional[Dict]], float]] = {
    "bert_score": bert_score,
    "exact_match": exact_match,
    "f1_mera": f1_mera,
}


def get_scorer(metric: str) -> Callable[[str, str, Optional[Dict]], float]:
    if metric not in _SCORERS:
        raise ValueError(
            f"Unknown metric {metric!r}. Supported: {SUPPORTED_METRICS}"
        )
    return _SCORERS[metric]


def default_metric_for(dataset: str) -> str:
    """Return the default metric for the dataset (bert_score if unknown)."""
    return DEFAULT_METRIC_BY_DATASET.get(dataset, "bert_score")
