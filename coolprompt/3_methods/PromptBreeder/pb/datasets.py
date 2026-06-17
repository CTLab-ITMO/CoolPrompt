"""Unified loader and fitness heuristics for the 8 datasets shipped under
``pb/data/``.

Every dataset uses the same on-disk schema::

    {"input": "...", "target": "..."}

so a single generic loader works for all of them. The per-dataset
correctness checks are intentionally lightweight, mirroring the original
GSM-style ``re.search`` heuristic used by the repository.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List


SUPPORTED_DATASETS: List[str] = [
    "squad_v2",
    "gsm8k",
    "common_gen",
    "xsum",
    "tweeteval",
    "mediqa",
    "code_to_text",
    "concode",
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# --------------------------------------------------------------------- loading
def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_examples(name: str, split: str = "train") -> List[Dict]:
    """Load a dataset split.

    Args:
        name: One of :data:`SUPPORTED_DATASETS`.
        split: ``"train"``, ``"validation"``, or ``"test"``.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {SUPPORTED_DATASETS}"
        )
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split {split!r}")
    path = os.path.join(_DATA_DIR, name, f"{split}_{name}.jsonl")
    return _read_jsonl(path)


# --------------------------------------------------------- default task prompts
_DEFAULT_PROBLEMS: Dict[str, str] = {
    "squad_v2": (
        "Given a passage and a question, extract the shortest exact answer "
        "span from the passage. If the answer is not in the passage, reply "
        "with 'unanswerable'."
    ),
    "gsm8k": (
        "Solve the following grade-school math word problem. Show your "
        "reasoning briefly and end your answer with the final numeric "
        "result on its own line."
    ),
    "common_gen": (
        "Given a list of concept words, write a single short, fluent "
        "English sentence that naturally uses ALL of the given concepts."
    ),
    "xsum": (
        "Summarize the following news article in a single concise sentence "
        "that captures its main point."
    ),
    "tweeteval": (
        "Classify the emotion expressed in the following tweet. Reply with "
        "exactly one lowercase label from: anger, joy, optimism, sadness."
    ),
    "mediqa": (
        "Given a consumer health question together with relevant medical "
        "reference passages, write a concise, factually grounded answer "
        "for the patient."
    ),
    "code_to_text": (
        "Given a Python function, write a concise and accurate docstring "
        "describing its behavior, arguments, and return value."
    ),
    "concode": (
        "Given a natural-language description of a Java method together "
        "with the surrounding class signature, generate the corresponding "
        "Java method body."
    ),
}


def default_problem_description(name: str) -> str:
    """Return a reasonable starting task description for the given dataset."""
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}")
    return _DEFAULT_PROBLEMS[name]


# ----------------------------------------------------------------- fitness ops
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "is",
    "be", "this", "that", "with", "as", "by", "it", "its", "if", "are",
    "was", "were", "from", "at", "but", "not", "can", "will", "we", "you",
    "i", "he", "she", "they", "them", "their", "his", "her", "our", "us",
    "do", "does", "did", "have", "has", "had", "so", "than", "then",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _keyword_overlap_correct(output: str, target: str, max_keywords: int = 5) -> bool:
    """Loose keyword-overlap check used for free-form text targets.

    A response is considered correct if the model output contains every
    keyword (non-stopword, length > 3) extracted from the first sentence
    of the reference.
    """
    if not target:
        return bool((output or "").strip())
    first_sentence = re.split(r"(?<=[.!?])\s", target.strip(), maxsplit=1)[0]
    tokens = _tokenize(first_sentence)
    keywords: List[str] = []
    for tok in tokens:
        if len(tok) <= 3 or tok in _STOPWORDS or tok in keywords:
            continue
        keywords.append(tok)
        if len(keywords) >= max_keywords:
            break
    if not keywords:
        return bool((output or "").strip())
    out_low = (output or "").lower()
    return all(kw in out_low for kw in keywords)


def _gsm_extract_number(text: str) -> str:
    """Pull the last numeric token from a string (after stripping commas)."""
    cleaned = (text or "").replace(",", "")
    matches = _NUMBER_RE.findall(cleaned)
    return matches[-1] if matches else ""


def _squad_correct(output: str, target: str) -> bool:
    target_norm = (target or "").strip().lower()
    if not target_norm:
        return bool((output or "").strip())
    return target_norm in (output or "").lower()


def _gsm_correct(output: str, target: str) -> bool:
    tgt = _gsm_extract_number(target)
    out = _gsm_extract_number(output)
    if not tgt:
        return False
    try:
        return float(tgt) == float(out)
    except (ValueError, TypeError):
        return False


def _common_gen_correct(output: str, input_text: str, target: str) -> bool:
    """All concepts from the input list must appear in the output."""
    concepts: List[str] = []
    # ``input`` looks like "['ski', 'mountain', 'skier']" — pull tokens.
    for tok in _TOKEN_RE.findall(input_text or ""):
        low = tok.lower()
        if low in _STOPWORDS or len(low) < 2:
            continue
        if low not in concepts:
            concepts.append(low)
    if not concepts:
        return _keyword_overlap_correct(output, target)
    out_low = (output or "").lower()
    return all(c in out_low for c in concepts)


def _tweeteval_correct(output: str, target: str) -> bool:
    tgt = (target or "").strip().lower()
    if not tgt:
        return False
    return re.search(rf"\b{re.escape(tgt)}\b", (output or "").lower()) is not None


def _concode_correct(output: str, target: str) -> bool:
    """Token-overlap on identifiers (Concode targets are short Java snippets)."""
    tgt_tokens = [
        t for t in _tokenize(target)
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tgt_tokens:
        return bool((output or "").strip())
    # Keep it cheap: require at least ~60% of the reference identifiers.
    out_low = (output or "").lower()
    hits = sum(1 for t in tgt_tokens if t in out_low)
    return hits / max(1, len(tgt_tokens)) >= 0.6


def score(name: str, metric: str, model_output: str, example: Dict) -> float:
    """Evaluate ``model_output`` against ``example`` using the named metric.

    Returns a float in ``[0, 1]``. The ``example`` dict is annotated with
    ``dataset`` so metric implementations can apply dataset-specific tweaks
    (e.g. numeric extraction for GSM8K exact-match).
    """
    from pb.metrics import get_scorer

    target = example.get("target", "")
    enriched = dict(example)
    enriched["dataset"] = name
    return float(get_scorer(metric)(model_output or "", target or "", enriched))


def is_correct(name: str, model_output: str, example: Dict) -> bool:
    """Dataset-aware correctness check used by the fitness evaluation."""
    target = example.get("target", "")
    input_text = example.get("input", "")
    if name == "gsm8k":
        return _gsm_correct(model_output, target)
    if name == "squad_v2":
        return _squad_correct(model_output, target)
    if name == "common_gen":
        return _common_gen_correct(model_output, input_text, target)
    if name == "tweeteval":
        return _tweeteval_correct(model_output, target)
    if name == "concode":
        return _concode_correct(model_output, target)
    # xsum, mediqa, code_to_text -> free-form keyword overlap
    return _keyword_overlap_correct(model_output, target)


# ----------------------------------------------------------------- eval prompt
def build_eval_prompt(task_prompt: str, example: Dict) -> str:
    """Compose the evaluation prompt fed to the LLM during fitness scoring."""
    return (
        f"{task_prompt}\n\n"
        f"INPUT:\n{example.get('input', '')}\n\n"
        f"ANSWER:"
    )
