"""Generic evaluator for the 8 datasets shipped under ``datasets/data/``.

Each dataset has a ``train_<name>.json`` and ``validation_<name>.json`` file
containing ``{"examples": [{"input": ..., "target": ...}, ...]}``.

The :func:`eval_dataset` function applies a candidate prompt (the optimised
"chain-of-thought" / instruction string) to a list of examples by calling the
shared LLM client and computes a dataset-specific metric in ``[0, 1]``.
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List

from tqdm import tqdm

from llm_client import turbo_query

DATASETS = [
    "squad_v2",
    "gsm8k",
    "common_gen",
    "xsum",
    "tweeteval",
    "mediqa",
    "code_to_text",
    "concode",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")


def load_split(dataset: str, split: str) -> List[Dict[str, str]]:
    """Load the ``train`` or ``validation`` split for a dataset."""
    path = os.path.join(_data_dir(), dataset, f"{split}_{dataset}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if examples is None:
        raise ValueError(f"Unexpected layout for {path}")
    return examples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

INSTRUCTIONS: Dict[str, str] = {
    "squad_v2": "Read the passage and answer the question with a short span from the passage.",
    "gsm8k": "Solve the following grade-school math word problem. End your response with 'The answer is <number>'.",
    "common_gen": "Write one short, fluent sentence that uses all of the given concepts in a natural way.",
    "xsum": "Summarise the following article in a single concise sentence.",
    "tweeteval": "Classify the emotion expressed in the tweet. Reply with a single lowercase word (e.g. anger, joy, optimism, sadness).",
    "mediqa": "Summarise the medical passage in one or two short sentences.",
    "code_to_text": "Write a brief natural-language description (docstring) for the following code snippet.",
    "concode": "Generate the Java method that implements the requested functionality. Reply with code only.",
}


def build_prompt(dataset: str, instruction: str, example: Dict[str, str]) -> str:
    """Combine the dataset-specific instruction, the candidate prompt and the
    actual example into the final string sent to the chat model."""
    base = INSTRUCTIONS.get(dataset, "Follow the instruction below.")
    inp = example.get("input", "")
    return (
        f"{base}\n"
        f"Guidance: {instruction}\n\n"
        f"Input:\n{inp}\n\n"
        f"Output:"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def squad_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Number that follows an explicit "answer is" / "answer:" cue, if present.
_ANSWER_CUE_RE = re.compile(
    r"(?:answer|result|total)\s*(?:is|:|=)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_last_number(text: str) -> str | None:
    cleaned = text.replace(",", "")
    matches = _NUM_RE.findall(cleaned)
    return matches[-1] if matches else None


def _extract_number_answer(text: str) -> str | None:
    """Extract the numeric answer from free-form text.

    Prefers a number that follows an explicit cue such as "The answer is 72",
    otherwise falls back to the last number in the text. The result is
    canonicalised to a plain number string (commas removed)."""
    if not text:
        return None
    cue_matches = _ANSWER_CUE_RE.findall(text)
    if cue_matches:
        return cue_matches[-1].replace(",", "")
    return _extract_last_number(text)


def _canonical_number(value: str | None) -> str | None:
    """Normalise a number string so that ``5``, ``5.0`` and ``05`` compare equal."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f.is_integer():
        return str(int(f))
    return repr(f)


def gsm8k_em(pred: str, gold: str) -> float:
    g = _extract_last_number(gold)
    p = _extract_last_number(pred)
    if g is None or p is None:
        return 0.0
    try:
        return 1.0 if float(p) == float(g) else 0.0
    except ValueError:
        return 0.0


# Emotion labels used by the tweeteval (emotion) dataset.
TWEETEVAL_LABELS = ["anger", "joy", "optimism", "sadness"]

# Morphological variants / common synonyms that the model produces instead of
# the exact canonical label. Each maps to one of ``TWEETEVAL_LABELS``. This is a
# metric-side answer-normalisation step (the task uses accuracy), so it does not
# change the EvoPrompt search; it only ensures free-form replies such as
# "angry", "happy" or "optimistic" are scored against the right class.
TWEETEVAL_SYNONYMS: Dict[str, str] = {
    # anger
    "anger": "anger", "angry": "anger", "angered": "anger", "mad": "anger",
    "furious": "anger", "fury": "anger", "rage": "anger", "enraged": "anger",
    "irritated": "anger", "annoyed": "anger", "outrage": "anger",
    "outraged": "anger", "hostile": "anger", "hate": "anger", "hatred": "anger",
    # joy
    "joy": "joy", "joyful": "joy", "joyous": "joy", "happy": "joy",
    "happiness": "joy", "glad": "joy", "delight": "joy", "delighted": "joy",
    "cheerful": "joy", "excited": "joy", "excitement": "joy", "elated": "joy",
    "pleased": "joy", "love": "joy", "amused": "joy",
    # optimism
    "optimism": "optimism", "optimistic": "optimism", "hope": "optimism",
    "hopeful": "optimism", "positive": "optimism", "positivity": "optimism",
    "confident": "optimism", "encouraged": "optimism", "encouraging": "optimism",
    # sadness
    "sadness": "sadness", "sad": "sadness", "sorrow": "sadness",
    "unhappy": "sadness", "depressed": "sadness", "depression": "sadness",
    "depressing": "sadness", "grief": "sadness", "miserable": "sadness",
    "misery": "sadness", "gloomy": "sadness", "melancholy": "sadness",
    "disappointed": "sadness", "down": "sadness",
}

# Canonical prefixes used as a last structured attempt before the raw fallback.
_TWEETEVAL_PREFIXES = {
    "anger": "anger", "angr": "anger",
    "joy": "joy", "joyf": "joy",
    "optimis": "optimism", "optimiz": "optimism",
    "sad": "sadness",
}


def _extract_tweeteval_label(text: str) -> str:
    """Return the canonical emotion label expressed in ``text``.

    The model often answers with a full sentence ("This tweet expresses
    optimism.") or a morphological variant / synonym ("angry", "happy",
    "optimistic") instead of the bare gold label. We resolve, in order:

    1. an exact canonical label (``anger``/``joy``/``optimism``/``sadness``);
    2. a known synonym or inflected form mapped back to its canonical label;
    3. a canonical prefix/stem match (e.g. ``optimistic`` -> ``optimism``);
    4. the first alphabetic word as a last-resort fallback.
    """
    if not text:
        return ""
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)

    # 1. Exact canonical label, earliest occurrence wins.
    best_label = ""
    best_pos = len(lowered) + 1
    for label in TWEETEVAL_LABELS:
        m = re.search(rf"\b{re.escape(label)}\b", lowered)
        if m is not None and m.start() < best_pos:
            best_pos = m.start()
            best_label = label
    if best_label:
        return best_label

    # 2. Synonym / inflected form: first matching word in reading order.
    for word in words:
        canonical = TWEETEVAL_SYNONYMS.get(word)
        if canonical:
            return canonical

    # 3. Canonical prefix / stem match.
    for word in words:
        for prefix, canonical in _TWEETEVAL_PREFIXES.items():
            if word.startswith(prefix):
                return canonical

    # 4. Fall back to the first alphabetic word in the response.
    return words[0] if words else ""


def extract_answer(dataset: str, text: str) -> str:
    """Dataset-aware extraction of the final answer from a model response.

    For most datasets the raw text is returned unchanged. For ``gsm8k`` we
    extract the numeric answer; for ``tweeteval`` we extract the emotion label.
    This keeps each dataset on its configured metric while ensuring the metric
    operates on comparable, cleaned answers rather than verbose free text."""
    if dataset == "gsm8k":
        num = _extract_number_answer(text)
        return _canonical_number(num) or ""
    if dataset == "tweeteval":
        return _extract_tweeteval_label(text)
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def f1_mera(pred: str, gold: str) -> float:
    """Token-level F1 score ("F1-мера") used as the default metric for tweeteval.

    Tokens are lowercased, stripped of punctuation/articles via ``_normalize_text``
    and matched as a multiset. This is the same definition that is used by
    SQuAD-style F1 and is well-defined both for short class labels (where it
    reduces to exact match) and for free-form text.
    """
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BERTScore (optional heavy dependency)
# ---------------------------------------------------------------------------

_BERT_SCORE_FN = None  # cached callable or False if unavailable
_BERT_SCORE_WARNED = False


def _get_bert_score_fn(logger: Any = None):
    """Lazily import ``bert_score`` and return its scoring function.

    Returns ``None`` if the package is not installed; callers should then fall
    back to a cheaper proxy metric (we use ROUGE-L F1) and log a warning once.
    """
    global _BERT_SCORE_FN, _BERT_SCORE_WARNED
    if _BERT_SCORE_FN is not None:
        return _BERT_SCORE_FN or None
    try:
        from bert_score import score as _bs_score  # type: ignore
        _BERT_SCORE_FN = _bs_score
        return _BERT_SCORE_FN
    except Exception as exc:  # noqa: BLE001
        _BERT_SCORE_FN = False  # sentinel: tried and failed
        if not _BERT_SCORE_WARNED:
            _BERT_SCORE_WARNED = True
            msg = (
                "bert-score is not installed or failed to import "
                f"({exc!r}); falling back to ROUGE-L F1 as a proxy. "
                "Install with `pip install bert-score` for the real metric."
            )
            if logger is not None:
                logger.warning(msg)
            else:
                print(f"[dataset_eval] WARNING: {msg}")
        return None


def bert_score_f1(pred: str, gold: str, logger: Any = None) -> float:
    """Single-pair BERTScore F1, falling back to ROUGE-L F1 if unavailable."""
    fn = _get_bert_score_fn(logger=logger)
    if fn is None:
        return rouge_l_f1(pred, gold)
    try:
        _, _, F1 = fn([pred or ""], [gold or ""], lang="en",
                      rescale_with_baseline=False, verbose=False)
        return float(F1[0].item())
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning(f"bert_score call failed ({exc!r}); using ROUGE-L F1.")
        return rouge_l_f1(pred, gold)


def _tokens(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    lcs = _lcs_length(p, g)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(g)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4(pred: str, gold: str) -> float:
    """Very small BLEU-4 implementation (corpus-level not needed for our use)."""
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    import math
    weights = [0.25] * 4
    log_p_sum = 0.0
    for n, w in enumerate(weights, 1):
        pred_ng = _ngrams(p, n)
        gold_ng = _ngrams(g, n)
        if not pred_ng:
            return 0.0
        overlap = sum(min(pred_ng[k], gold_ng[k]) for k in pred_ng)
        total = sum(pred_ng.values())
        if overlap == 0:
            return 0.0
        log_p_sum += w * math.log(overlap / total)
    # brevity penalty
    if len(p) > len(g):
        bp = 1.0
    else:
        bp = math.exp(1 - len(g) / max(len(p), 1))
    return bp * math.exp(log_p_sum)


# Three user-facing metrics that can be selected for optimisation.
METRIC_FUNCS: Dict[str, Callable[..., float]] = {
    "bert_score": bert_score_f1,
    "exact_match": exact_match,
    "f1_mera": f1_mera,
}

# Default metric for every dataset:
#   gsm8k     -> exact_match
#   tweeteval -> f1_mera
#   all other -> bert_score
DEFAULT_METRICS: Dict[str, str] = {
    "squad_v2":     "bert_score",
    "gsm8k":        "exact_match",
    "common_gen":   "bert_score",
    "xsum":         "bert_score",
    "tweeteval":    "f1_mera",
    "mediqa":       "bert_score",
    "code_to_text": "bert_score",
    "concode":      "bert_score",
}


def resolve_metric(dataset: str, metric: str | None = None) -> str:
    """Return a valid metric name from ``METRIC_FUNCS``.

    If ``metric`` is given, it is validated. Otherwise the default for the
    given dataset is returned.
    """
    if metric is not None:
        if metric not in METRIC_FUNCS:
            raise ValueError(
                f"Unknown metric {metric!r}; choices: {list(METRIC_FUNCS)}"
            )
        return metric
    if dataset in DEFAULT_METRICS:
        return DEFAULT_METRICS[dataset]
    # Sensible fallback for legacy / unknown datasets.
    return "bert_score"


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def eval_dataset(dataset: str, cot_prompt: str, eval_data: List[Dict[str, str]],
                 client: Any = None, model_index: str = "turbo",
                 logger: Any = None, demon: int = 0,
                 metric: str | None = None, **kwargs) -> float:
    """Evaluate a candidate prompt on ``eval_data`` and return the mean score.

    Parameters
    ----------
    metric:
        Optional metric name (one of ``bert_score``, ``exact_match``,
        ``f1_mera``). When ``None`` the per-dataset default from
        :data:`DEFAULT_METRICS` is used.
    """
    metric_name = resolve_metric(dataset, metric)
    metric_fn = METRIC_FUNCS[metric_name]

    def _score(pred: str, gold: str) -> float:
        if metric_name == "bert_score":
            return bert_score_f1(pred, gold, logger=logger)
        return metric_fn(pred, gold)

    scores: List[float] = []
    first = True
    for ex in tqdm(eval_data, desc=f"eval[{dataset}|{metric_name}]"):
        prompt = build_prompt(dataset, cot_prompt, ex)
        try:
            response = turbo_query(prompt)
        except Exception as exc:  # noqa: BLE001 - keep evolving even on errors
            if logger is not None:
                logger.warning(f"LLM call failed, scoring 0: {exc}")
            response = ""
        gold = str(ex.get("target", ""))
        # Dataset-aware extraction so the metric compares clean answers, not
        # the verbose free-form response (critical for gsm8k / tweeteval).
        pred_ans = extract_answer(dataset, response)
        gold_ans = extract_answer(dataset, gold)
        score = float(_score(pred_ans, gold_ans))
        scores.append(score)
        if first and logger is not None:
            logger.info(f"[{dataset}|{metric_name}] sample prompt:\n{prompt}")
            logger.info(f"[{dataset}|{metric_name}] sample response: {response}")
            logger.info(f"[{dataset}|{metric_name}] extracted pred: {pred_ans!r}")
            logger.info(f"[{dataset}|{metric_name}] gold: {gold} "
                        f"(extracted: {gold_ans!r})")
            logger.info(f"[{dataset}|{metric_name}] score: {score}")
            first = False

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
