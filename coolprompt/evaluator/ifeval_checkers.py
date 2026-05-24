"""Verifiable checkers for a subset of IFEval instructions.

Each checker returns True iff the model `response` satisfies the
instruction described by its `kwargs`. Implements a focused
subset of the IFEval (Zhou et al., 2023) instruction registry;
the loader filters the dataset to these ids so every evaluated
prompt is checkable.
"""

import json
import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    return len([p for p in parts if p.strip()])


def _check_relation(
    value: int, threshold: int, relation: str
) -> bool:
    if relation == "at least":
        return value >= threshold
    if relation == "at most":
        return value <= threshold
    if relation == "less than":
        return value < threshold
    if relation == "exactly":
        return value == threshold
    return False


def _number_words(resp, kw):
    return _check_relation(
        _word_count(resp),
        int(kw.get("num_words", 0)),
        kw.get("relation", "at least"),
    )


def _number_sentences(resp, kw):
    return _check_relation(
        _sentence_count(resp),
        int(kw.get("num_sentences", 0)),
        kw.get("relation", "at least"),
    )


def _keyword_existence(resp, kw):
    # Substring (not word-boundary) match is intentional, matching
    # the IFEval reference impl; differs from _keyword_frequency.
    low = resp.lower()
    return all(k.lower() in low for k in kw.get("keywords", []))


def _keyword_frequency(resp, kw):
    word = str(kw.get("keyword", "")).lower()
    count = len(
        re.findall(
            r"\b" + re.escape(word) + r"\b", resp.lower()
        )
    )
    return _check_relation(
        count,
        int(kw.get("frequency", 0)),
        kw.get("relation", "at least"),
    )


def _forbidden_words(resp, kw):
    low = resp.lower()
    return all(
        re.search(
            r"\b" + re.escape(w.lower()) + r"\b", low
        ) is None
        for w in kw.get("forbidden_words", [])
    )


def _all_lowercase(resp, kw):
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _all_uppercase(resp, kw):
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _number_bullets(resp, kw):
    bullets = re.findall(
        r"^\s*[\*\-]\s+", resp, re.MULTILINE
    )
    return len(bullets) == int(kw.get("num_bullets", 0))


def _json_format(resp, kw):
    text = resp.strip()
    text = re.sub(r"^```(json)?|```$", "", text).strip()
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _number_highlighted(resp, kw):
    highlights = re.findall(
        r"(?<!\*)\*(?!\*)[^\*\n]+?(?<!\*)\*(?!\*)", resp
    )
    return len(highlights) >= int(
        kw.get("num_highlights", 0)
    )


def _title(resp, kw):
    return re.search(r"<<[^>\n]+>>", resp) is not None


def _end_checker(resp, kw):
    phrase = str(kw.get("end_phrase", "")).strip().lower()
    return resp.strip().lower().endswith(phrase)


def _quotation(resp, kw):
    t = resp.strip()
    return len(t) >= 2 and t.startswith('"') and t.endswith('"')


def _postscript(resp, kw):
    marker = str(
        kw.get("postscript_marker", "P.S.")
    ).lower()
    return marker in resp.lower()


def _placeholders(resp, kw):
    found = re.findall(r"\[[^\]\n]+\](?!\()", resp)
    return len(found) >= int(
        kw.get("num_placeholders", 0)
    )


# instruction id -> checker function
_CHECKERS = {
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "keywords:existence": _keyword_existence,
    "keywords:frequency": _keyword_frequency,
    "keywords:forbidden_words": _forbidden_words,
    "change_case:english_lowercase": _all_lowercase,
    "change_case:english_capital": _all_uppercase,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:json_format": _json_format,
    "detectable_format:number_highlighted_sections": (
        _number_highlighted
    ),
    "detectable_format:title": _title,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "detectable_content:postscript": _postscript,
    "detectable_content:number_placeholders": _placeholders,
}

SUPPORTED_INSTRUCTIONS = frozenset(_CHECKERS.keys())


def check_instruction(
    instruction_id: str, response: str, kwargs: dict
) -> bool:
    """Return True iff `response` satisfies the instruction.

    Unknown instruction ids return False (treated as a failed
    constraint), so unsupported rows must be filtered upstream.
    """
    checker = _CHECKERS.get(instruction_id)
    if checker is None:
        return False
    try:
        return bool(checker(response, kwargs or {}))
    except (ValueError, TypeError, re.error):
        return False
