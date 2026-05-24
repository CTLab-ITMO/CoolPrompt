# Thesis Benchmarks + Local Model Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new benchmark families (IFEval, GSM8K+SVAMP, classic text classification, a Russian task) and a swappable local model ladder (LM Studio) to the CoolPrompt PE2/SGR comparison harness, so the thesis can report SGR vs PE2/OPRO/APE across tasks and model scales.

**Architecture:** New benchmarks are added as (a) dataset loaders under `src/utils/` returning `pd.DataFrame[input_data, target]` (matching `load_dataset_pe2_paper.py`), and (b) where a new scoring rule is needed, a `GenerationMetric`/`ClassificationMetric` subclass in `coolprompt/evaluator/metrics.py` (auto-registered via `__subclasses__()`). A new benchmark config module wires them into the existing `ParallelBenchmarkRunner`/`BenchmarkTask` flow. Model selection is extracted out of the hardcoded `ChatOpenAI(gpt-4o-mini)` into a `make_llm()` factory pointed at LM Studio's OpenAI-compatible server (`http://localhost:1234/v1`), with an Anthropic cross-family slot gated behind an env key.

**Tech Stack:** Python 3.12, `uv`, LangChain (`langchain-openai`, `langchain-anthropic`), HuggingFace `datasets`/`evaluate`, `nltk`+`langdetect` (already installed), LM Studio (`lms`), `unittest` (repo test runner), `flake8`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/solutions/PE2/local_models.py` | `make_llm(name)` factory + `MODEL_LADDER`; LM Studio + Anthropic | Create |
| `src/utils/load_dataset_ifeval.py` | Load `google/IFEval`, filter to supported instruction types, encode constraint spec into `target` as JSON | Create |
| `coolprompt/evaluator/ifeval_checkers.py` | Pure functions: one verifiable checker per supported IFEval instruction id | Create |
| `coolprompt/evaluator/metrics.py` | Add `IFEvalMetric(GenerationMetric)` (name `ifeval`) using the checkers | Modify |
| `src/utils/load_dataset_math.py` | Load GSM8K + SVAMP into `input_data`/`target` (numeric answer) | Create |
| `src/utils/load_dataset_classification.py` | Load SST-2, AG News, TREC, Subj as classification | Create |
| `src/utils/load_dataset_russian.py` | Load RuCoLA (Russian acceptability) classification | Create |
| `src/solutions/PE2/extra_benchmarks_config.py` | Build `BenchmarkTask` configs for all new datasets | Create |
| `src/solutions/PE2/extra_benchmarks_test.py` | CLI entrypoint: pick benchmark + model, run all 4 methods | Create |
| `test/coolprompt/evaluator/test_ifeval_checkers.py` | Unit tests for the checker functions + metric | Create |
| `test/utils/test_new_loaders.py` | Smoke tests: each loader returns non-empty `[input_data, target]` | Create |

**Conventions to follow (verified in repo):**
- Always run Python via `uv run` (never bare `python`/`pip`) — per `CLAUDE.md`.
- Loaders return `pd.DataFrame` with exactly columns `input_data` (str) and `target` (str).
- Metrics auto-register from `GenerationMetric.__subclasses__()` / `ClassificationMetric.__subclasses__()` — just subclass and define `_get_name()`.
- `GenerationMetric.compute()` extracts text between `<ans>`/`</ans>` tags before scoring; IFEval must bypass this (score the raw output), so override `compute()`.
- Classification metrics expect the model to emit a label inside `<ans>...</ans>`; targets are arbitrary strings encoded to ids by `extract_labels`.
- Lint: `uv run flake8 <files>` must pass (repo uses flake8; lines kept ≤ ~79 cols in `metrics.py`).
- Tests: `uv run python -m unittest <module>`.

---

## Task 1: Local model ladder factory

**Files:**
- Create: `src/solutions/PE2/local_models.py`
- Modify: `src/solutions/PE2/sgr_advantage_test.py:135-140` (replace hardcoded `ChatOpenAI`)

- [ ] **Step 1: Create the model factory**

LM Studio serves an OpenAI-compatible API at `http://localhost:1234/v1`. The model id must match a model loaded in LM Studio (check with `lms ls`). The ladder uses three scales that fit in 48 GB unified memory.

Create `src/solutions/PE2/local_models.py`:

```python
"""Model factory for benchmark runs.

Local models are served by LM Studio's OpenAI-compatible
server (default http://localhost:1234/v1). Load the target
model in LM Studio first (`lms load <model>`), then pass its
id via --model.

A cross-family Anthropic slot is available when CP_ANTHROPIC_KEY
is set; it is part of the thesis design but execution is
local-only for now.
"""

import os

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

LMSTUDIO_BASE_URL = os.environ.get(
    "CP_LMSTUDIO_URL", "http://localhost:1234/v1"
)

# Logical ladder name -> LM Studio model id.
# Adjust ids to match `lms ls` output on this machine.
MODEL_LADDER = {
    "weak": "qwen3-1.7b",
    "mid": "qwen3-4b-instruct-2507",
    "strong": "qwen3-8b",
}

# Cross-family API slot (design only; needs CP_ANTHROPIC_KEY).
ANTHROPIC_MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
}


def make_llm(
    name: str,
    temperature: float = 0.0,
    max_retries: int = 10,
    request_timeout: int = 600,
) -> BaseLanguageModel:
    """Build a LangChain chat model for a benchmark run.

    Args:
        name: A MODEL_LADDER key, an ANTHROPIC_MODELS key, or a
            raw LM Studio model id.
        temperature: Sampling temperature.
        max_retries: Client-side retry count.
        request_timeout: Per-request timeout (seconds); local
            models can be slow, so default is generous.

    Returns:
        A configured BaseLanguageModel.
    """
    if name in ANTHROPIC_MODELS:
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("CP_ANTHROPIC_KEY")
        if not api_key:
            raise RuntimeError(
                "CP_ANTHROPIC_KEY not set for Anthropic model "
                f"'{name}'"
            )
        return ChatAnthropic(
            model=ANTHROPIC_MODELS[name],
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=request_timeout,
        )

    model_id = MODEL_LADDER.get(name, name)
    return ChatOpenAI(
        model=model_id,
        base_url=LMSTUDIO_BASE_URL,
        api_key=os.environ.get("CP_LMSTUDIO_KEY", "lm-studio"),
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )
```

- [ ] **Step 2: Lint the new module**

Run: `uv run flake8 src/solutions/PE2/local_models.py`
Expected: no output (clean).

- [ ] **Step 3: Verify LM Studio connectivity**

Start the server and load the mid model first:
```bash
lms server start
lms load qwen3-4b-instruct-2507   # or whatever `lms ls` shows
```
Then:
```bash
uv run python -c "from src.solutions.PE2.local_models import make_llm; m=make_llm('mid'); print(m.invoke('Reply with the single word: ok').content)"
```
Expected: prints a response containing `ok`. If connection refused, the LM Studio server is not running.

- [ ] **Step 4: Wire factory into existing SGR test (optional model flag)**

In `src/solutions/PE2/sgr_advantage_test.py`, replace the hardcoded block at lines 135-140:

```python
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("CP_OPENAI_KEY"),
        max_retries=10,
        request_timeout=120,
    )
```

with:

```python
    from src.solutions.PE2.local_models import make_llm
    llm = make_llm(model_name)
```

Add `model_name: str = "mid"` to the `run_sgr_advantage(...)` signature and a CLI arg in the argparse block (mirror the existing `--method` arg):

```python
    parser.add_argument(
        "--model", default="mid",
        help="MODEL_LADDER key or LM Studio model id",
    )
```
and pass `model_name=args.model` into `run_sgr_advantage(...)`.

- [ ] **Step 5: Lint and commit**

Run: `uv run flake8 src/solutions/PE2/local_models.py src/solutions/PE2/sgr_advantage_test.py`
Expected: clean.

```bash
git add src/solutions/PE2/local_models.py src/solutions/PE2/sgr_advantage_test.py
git commit -m "feat(bench): add local model ladder factory (LM Studio + Anthropic slot)"
```

---

## Task 2: IFEval checker functions (TDD)

**Files:**
- Create: `coolprompt/evaluator/ifeval_checkers.py`
- Test: `test/coolprompt/evaluator/test_ifeval_checkers.py`

We implement a focused, fully-specified subset of IFEval's verifiable instruction types. The loader (Task 3) filters the dataset to rows whose `instruction_id_list` is a subset of `SUPPORTED_INSTRUCTIONS`, so every loaded row is checkable. Each checker returns `bool` (constraint satisfied) given the model `response` and the instruction's `kwargs`.

- [ ] **Step 1: Write the failing tests**

Create `test/coolprompt/evaluator/test_ifeval_checkers.py`:

```python
import unittest

from coolprompt.evaluator.ifeval_checkers import (
    check_instruction,
    SUPPORTED_INSTRUCTIONS,
)


class TestIFEvalCheckers(unittest.TestCase):
    def test_number_words_at_least(self):
        resp = "one two three four five six"
        ok = check_instruction(
            "length_constraints:number_words",
            resp,
            {"num_words": 5, "relation": "at least"},
        )
        self.assertTrue(ok)

    def test_number_words_at_least_fail(self):
        ok = check_instruction(
            "length_constraints:number_words",
            "too short",
            {"num_words": 5, "relation": "at least"},
        )
        self.assertFalse(ok)

    def test_keyword_existence(self):
        ok = check_instruction(
            "keywords:existence",
            "the quick brown fox",
            {"keywords": ["quick", "fox"]},
        )
        self.assertTrue(ok)

    def test_forbidden_words(self):
        ok = check_instruction(
            "keywords:forbidden_words",
            "a clean sentence",
            {"forbidden_words": ["banned"]},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "keywords:forbidden_words",
            "this is banned",
            {"forbidden_words": ["banned"]},
        )
        self.assertFalse(bad)

    def test_all_lowercase(self):
        ok = check_instruction(
            "change_case:english_lowercase",
            "all lower case here",
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "change_case:english_lowercase",
            "Has Uppercase",
            {},
        )
        self.assertFalse(bad)

    def test_number_bullets(self):
        resp = "* one\n* two\n* three"
        ok = check_instruction(
            "detectable_format:number_bullet_lists",
            resp,
            {"num_bullets": 3},
        )
        self.assertTrue(ok)

    def test_json_format(self):
        ok = check_instruction(
            "detectable_format:json_format",
            '{"a": 1, "b": 2}',
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "detectable_format:json_format",
            "not json",
            {},
        )
        self.assertFalse(bad)

    def test_end_checker(self):
        ok = check_instruction(
            "startend:end_checker",
            "blah blah That is all.",
            {"end_phrase": "That is all."},
        )
        self.assertTrue(ok)

    def test_unknown_instruction_returns_false(self):
        ok = check_instruction("nonexistent:thing", "x", {})
        self.assertFalse(ok)

    def test_supported_set_nonempty(self):
        self.assertIn(
            "keywords:existence", SUPPORTED_INSTRUCTIONS
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest test.coolprompt.evaluator.test_ifeval_checkers -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: coolprompt.evaluator.ifeval_checkers`.

- [ ] **Step 3: Implement the checkers**

Create `coolprompt/evaluator/ifeval_checkers.py`:

```python
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


def _check_relation(value: int, threshold: int, relation: str) -> bool:
    if relation == "at least":
        return value >= threshold
    if relation == "at most":
        return value <= threshold
    if relation == "less than":
        return value < threshold
    if relation == "exactly":
        return value == threshold
    return value >= threshold


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
        re.search(r"\b" + re.escape(w.lower()) + r"\b", low) is None
        for w in kw.get("forbidden_words", [])
    )


def _all_lowercase(resp, kw):
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _all_uppercase(resp, kw):
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _number_bullets(resp, kw):
    bullets = re.findall(r"^\s*[\*\-]\s+", resp, re.MULTILINE)
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
    highlights = re.findall(r"\*[^\*\n]+\*", resp)
    return len(highlights) >= int(kw.get("num_highlights", 0))


def _title(resp, kw):
    return re.search(r"<<[^>\n]+>>", resp) is not None


def _end_checker(resp, kw):
    phrase = str(kw.get("end_phrase", "")).strip().lower()
    return resp.strip().lower().endswith(phrase)


def _quotation(resp, kw):
    t = resp.strip()
    return len(t) >= 2 and t.startswith('"') and t.endswith('"')


def _postscript(resp, kw):
    marker = str(kw.get("postscript_marker", "P.S.")).lower()
    return marker in resp.lower()


def _placeholders(resp, kw):
    found = re.findall(r"\[[^\]\n]+\]", resp)
    return len(found) >= int(kw.get("num_placeholders", 0))


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest test.coolprompt.evaluator.test_ifeval_checkers -v`
Expected: all tests PASS.

- [ ] **Step 5: Lint and commit**

Run: `uv run flake8 coolprompt/evaluator/ifeval_checkers.py test/coolprompt/evaluator/test_ifeval_checkers.py`
Expected: clean.

```bash
git add coolprompt/evaluator/ifeval_checkers.py test/coolprompt/evaluator/test_ifeval_checkers.py
git commit -m "feat(metrics): add IFEval verifiable instruction checkers"
```

---

## Task 3: IFEval metric + loader (TDD for metric)

**Files:**
- Modify: `coolprompt/evaluator/metrics.py` (add `IFEvalMetric`)
- Create: `src/utils/load_dataset_ifeval.py`
- Test: extend `test/coolprompt/evaluator/test_ifeval_checkers.py`

The metric reports IFEval **prompt-level strict accuracy**: a prompt scores 1.0 only if ALL its instructions are satisfied. The per-prompt constraint spec is JSON encoded into `target` by the loader as `{"instruction_id_list": [...], "kwargs": [{...}, ...]}`. IFEval must score the **raw** model output, so `compute()` is overridden to skip `<ans>` extraction.

- [ ] **Step 1: Write the failing metric test**

Append to `test/coolprompt/evaluator/test_ifeval_checkers.py`:

```python
import json as _json

from coolprompt.evaluator.metrics import IFEvalMetric


class TestIFEvalMetric(unittest.TestCase):
    def _spec(self, ids, kwargs_list):
        return _json.dumps(
            {"instruction_id_list": ids, "kwargs": kwargs_list}
        )

    def test_all_constraints_satisfied_scores_one(self):
        target = self._spec(
            ["keywords:existence",
             "change_case:english_lowercase"],
            [{"keywords": ["fox"]}, {}],
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["the quick brown fox"],
            targets=[target],
            dataset=["write about a fox"],
        )
        self.assertEqual(score, 1.0)

    def test_one_constraint_failed_scores_zero(self):
        target = self._spec(
            ["keywords:existence",
             "change_case:english_lowercase"],
            [{"keywords": ["fox"]}, {}],
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["The Quick Brown FOX"],
            targets=[target],
            dataset=["write about a fox"],
        )
        self.assertEqual(score, 0.0)

    def test_mean_over_prompts(self):
        t1 = self._spec(
            ["keywords:existence"], [{"keywords": ["a"]}]
        )
        t2 = self._spec(
            ["keywords:existence"], [{"keywords": ["zzz"]}]
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["a", "a"],
            targets=[t1, t2],
            dataset=["", ""],
        )
        self.assertEqual(score, 0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m unittest test.coolprompt.evaluator.test_ifeval_checkers.TestIFEvalMetric -v`
Expected: FAIL — `ImportError: cannot import name 'IFEvalMetric'`.

- [ ] **Step 3: Implement `IFEvalMetric` in `metrics.py`**

Add this import near the top of `coolprompt/evaluator/metrics.py` (after the existing `import re`):

```python
import json
from coolprompt.evaluator.ifeval_checkers import check_instruction
```

Add this class immediately after `ExactMatchMetric` (around line 464):

```python
class IFEvalMetric(GenerationMetric):
    """IFEval prompt-level strict accuracy.

    `targets` are JSON specs:
        {"instruction_id_list": [...], "kwargs": [{...}, ...]}
    A prompt scores 1.0 iff every listed instruction is
    satisfied by the raw model output; the metric returns the
    mean over prompts. Unlike other generation metrics, this
    scores the raw output (no <ans> tag extraction).
    """

    FORMAT_MISMATCH_LABEL = ""

    @staticmethod
    def _get_name():
        return "ifeval"

    def __init__(self):
        super().__init__()

    def compute(self, outputs, targets, dataset=None):
        return self._compute_raw(
            list(map(str, outputs)),
            list(map(str, targets)),
            dataset,
        )

    def _compute_raw(self, outputs, targets, dataset):
        per_prompt = []
        for output, spec_json in zip(outputs, targets):
            try:
                spec = json.loads(spec_json)
            except (ValueError, TypeError):
                per_prompt.append(0.0)
                continue
            ids = spec.get("instruction_id_list", [])
            kw_list = spec.get("kwargs", [{}] * len(ids))
            satisfied = all(
                check_instruction(iid, output, kw or {})
                for iid, kw in zip(ids, kw_list)
            )
            per_prompt.append(1.0 if satisfied and ids else 0.0)
        return float(mean(per_prompt)) if per_prompt else 0.0
```

Note: `mean` is already imported in `metrics.py` from `coolprompt.utils.arithmetics`. The class auto-registers in `GENERATION_METRIC_NAME_MAPPING` because it subclasses `GenerationMetric`.

- [ ] **Step 4: Run metric tests to verify pass**

Run: `uv run python -m unittest test.coolprompt.evaluator.test_ifeval_checkers -v`
Expected: all PASS.

- [ ] **Step 5: Verify metric is registered**

Run:
```bash
uv run python -c "from coolprompt.evaluator.metrics import GENERATION_METRIC_NAME_MAPPING as m; print('ifeval' in m)"
```
Expected: `True`.

- [ ] **Step 6: Implement the IFEval loader**

Create `src/utils/load_dataset_ifeval.py`:

```python
"""Loader for the IFEval benchmark (google/IFEval).

Filters to prompts whose instruction types are all supported by
coolprompt's verifiable checkers, and encodes each prompt's
constraint spec into `target` as JSON for IFEvalMetric.
"""

import json

import pandas as pd
from datasets import load_dataset

from coolprompt.evaluator.ifeval_checkers import (
    SUPPORTED_INSTRUCTIONS,
)


def load_ifeval(max_rows: int | None = None) -> pd.DataFrame:
    """Load IFEval prompts checkable by coolprompt.

    Args:
        max_rows: Optional cap on number of rows returned.

    Returns:
        DataFrame with `input_data` (the prompt) and `target`
        (JSON: instruction_id_list + kwargs).
    """
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for ex in ds:
        ids = ex["instruction_id_list"]
        if not ids or not all(
            i in SUPPORTED_INSTRUCTIONS for i in ids
        ):
            continue
        kwargs = [
            {k: v for k, v in d.items() if v is not None}
            for d in ex["kwargs"]
        ]
        rows.append(
            {
                "input_data": str(ex["prompt"]),
                "target": json.dumps(
                    {
                        "instruction_id_list": ids,
                        "kwargs": kwargs,
                    }
                ),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=["input_data", "target"])
```

- [ ] **Step 7: Smoke-test the loader**

Run:
```bash
uv run python -c "from src.utils.load_dataset_ifeval import load_ifeval; df=load_ifeval(max_rows=20); print(len(df), df.columns.tolist()); print(df.iloc[0]['target'][:120])"
```
Expected: prints a count > 0, `['input_data', 'target']`, and a JSON snippet starting with `{"instruction_id_list"`.

- [ ] **Step 8: Lint and commit**

Run: `uv run flake8 coolprompt/evaluator/metrics.py src/utils/load_dataset_ifeval.py`
Expected: clean.

```bash
git add coolprompt/evaluator/metrics.py src/utils/load_dataset_ifeval.py test/coolprompt/evaluator/test_ifeval_checkers.py
git commit -m "feat(bench): add IFEval metric and loader"
```

---

## Task 4: Math loaders (GSM8K + SVAMP)

**Files:**
- Create: `src/utils/load_dataset_math.py`
- Test: covered by `test/utils/test_new_loaders.py` (Task 7)

These use the existing `em` (ExactMatchMetric) generation metric, which compares numbers via `extract_number_from_text`. The model is expected to emit its answer inside `<ans>...</ans>`; the start prompt (set in config, Task 8) instructs that.

- [ ] **Step 1: Implement the loader**

Create `src/utils/load_dataset_math.py`:

```python
"""Loaders for math word-problem benchmarks (GSM8K, SVAMP)."""

import re

import pandas as pd
from datasets import load_dataset


def _final_number(text: str) -> str:
    """Extract the final numeric answer from a GSM8K answer."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else text.strip()


def load_gsm8k(split: str = "test") -> pd.DataFrame:
    """Load GSM8K into input_data/target (numeric target)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return pd.DataFrame(
        {
            "input_data": [str(q) for q in ds["question"]],
            "target": [_final_number(a) for a in ds["answer"]],
        }
    )


def load_svamp(split: str = "test") -> pd.DataFrame:
    """Load SVAMP into input_data/target (numeric target)."""
    ds = load_dataset("ChilleD/SVAMP", split=split)
    questions = [
        f"{b} {q}".strip()
        for b, q in zip(ds["Body"], ds["Question"])
    ]
    answers = [
        str(a).rstrip("0").rstrip(".") if "." in str(a) else str(a)
        for a in ds["Answer"]
    ]
    return pd.DataFrame(
        {"input_data": questions, "target": answers}
    )
```

- [ ] **Step 2: Smoke-test both loaders**

Run:
```bash
uv run python -c "from src.utils.load_dataset_math import load_gsm8k, load_svamp; g=load_gsm8k(); s=load_svamp(); print('gsm8k', len(g), g.iloc[0]['target']); print('svamp', len(s), s.iloc[0]['target'])"
```
Expected: both counts > 0; targets are numeric strings.

- [ ] **Step 3: Lint and commit**

Run: `uv run flake8 src/utils/load_dataset_math.py`
Expected: clean.

```bash
git add src/utils/load_dataset_math.py
git commit -m "feat(bench): add GSM8K and SVAMP loaders"
```

---

## Task 5: Classic classification loaders

**Files:**
- Create: `src/utils/load_dataset_classification.py`

These are `task="classification"`, scored by the existing `accuracy`/`f1` metrics. Targets are human-readable label strings (e.g. `positive`); the config's start prompt (Task 8) tells the model to answer with a label inside `<ans>`.

- [ ] **Step 1: Implement the loaders**

Create `src/utils/load_dataset_classification.py`:

```python
"""Loaders for classic text-classification benchmarks.

SST-2 (sentiment), AG News (topic), TREC (question type),
SUBJ (subjectivity). All return input_data + a human-readable
label string in target.
"""

import pandas as pd
from datasets import load_dataset

SST2_LABELS = {0: "negative", 1: "positive"}
AGNEWS_LABELS = {
    0: "world", 1: "sports", 2: "business", 3: "technology",
}
TREC_LABELS = {
    0: "abbreviation", 1: "entity", 2: "description",
    3: "human", 4: "location", 5: "number",
}
SUBJ_LABELS = {0: "objective", 1: "subjective"}


def _frame(texts, labels, mapping):
    return pd.DataFrame(
        {
            "input_data": [str(t) for t in texts],
            "target": [mapping[int(lbl)] for lbl in labels],
        }
    )


def load_sst2(split: str = "validation") -> pd.DataFrame:
    ds = load_dataset("stanfordnlp/sst2", split=split)
    return _frame(ds["sentence"], ds["label"], SST2_LABELS)


def load_agnews(split: str = "test") -> pd.DataFrame:
    ds = load_dataset("fancyzhx/ag_news", split=split)
    return _frame(ds["text"], ds["label"], AGNEWS_LABELS)


def load_trec(split: str = "test") -> pd.DataFrame:
    ds = load_dataset("CogComp/trec", split=split)
    return _frame(
        ds["text"], ds["coarse_label"], TREC_LABELS
    )


def load_subj(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("SetFit/subj", split=split)
    return _frame(ds["text"], ds["label"], SUBJ_LABELS)
```

- [ ] **Step 2: Smoke-test the loaders**

Run:
```bash
uv run python -c "from src.utils.load_dataset_classification import load_sst2, load_agnews, load_trec, load_subj; [print(n, len(f), sorted(set(f['target']))) for n,f in [('sst2',load_sst2()),('agnews',load_agnews()),('trec',load_trec()),('subj',load_subj())]]"
```
Expected: each prints count > 0 and its label set. If a dataset id 404s, note the failure and pick the documented alternative id (HF hub ids drift); do not silently skip.

- [ ] **Step 3: Lint and commit**

Run: `uv run flake8 src/utils/load_dataset_classification.py`
Expected: clean.

```bash
git add src/utils/load_dataset_classification.py
git commit -m "feat(bench): add SST-2/AG News/TREC/Subj classification loaders"
```

---

## Task 6: Russian task loader (RuCoLA)

**Files:**
- Create: `src/utils/load_dataset_russian.py`

RuCoLA = Russian Corpus of Linguistic Acceptability (binary: acceptable / unacceptable). `task="classification"`, scored by `accuracy`/`f1`. `bertscore` (used elsewhere) already uses `bert-base-multilingual-cased`, so the stack handles Russian; classification needs no embeddings.

- [ ] **Step 1: Implement the loader**

Create `src/utils/load_dataset_russian.py`:

```python
"""Loader for RuCoLA (Russian linguistic acceptability)."""

import pandas as pd
from datasets import load_dataset

RUCOLA_LABELS = {0: "unacceptable", 1: "acceptable"}


def load_rucola(split: str = "validation") -> pd.DataFrame:
    """Load RuCoLA into input_data/target (label string)."""
    ds = load_dataset("RussianNLP/rucola", split=split)
    return pd.DataFrame(
        {
            "input_data": [str(s) for s in ds["sentence"]],
            "target": [
                RUCOLA_LABELS[int(lbl)] for lbl in ds["label"]
            ],
        }
    )
```

- [ ] **Step 2: Smoke-test the loader**

Run:
```bash
uv run python -c "from src.utils.load_dataset_russian import load_rucola; df=load_rucola(); print(len(df), sorted(set(df['target']))); print(df.iloc[0]['input_data'])"
```
Expected: count > 0, labels `['acceptable', 'unacceptable']`, a Russian sentence printed. If the id 404s, try `RussianNLP/RuCoLA`; report which worked.

- [ ] **Step 3: Lint and commit**

Run: `uv run flake8 src/utils/load_dataset_russian.py`
Expected: clean.

```bash
git add src/utils/load_dataset_russian.py
git commit -m "feat(bench): add RuCoLA Russian classification loader"
```

---

## Task 7: Loader smoke tests

**Files:**
- Create: `test/utils/test_new_loaders.py`

These tests guard the loader contract (non-empty, correct columns/types). They hit the network/HF cache, so they are slow; mark them clearly and keep `max_rows` small where supported.

- [ ] **Step 1: Write the tests**

Create `test/utils/test_new_loaders.py`:

```python
import unittest

import pandas as pd


def _assert_contract(df):
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["input_data", "target"]
    assert len(df) > 0
    assert df["input_data"].map(type).eq(str).all()
    assert df["target"].map(type).eq(str).all()


class TestNewLoaders(unittest.TestCase):
    def test_ifeval(self):
        from src.utils.load_dataset_ifeval import load_ifeval
        _assert_contract(load_ifeval(max_rows=10))

    def test_gsm8k(self):
        from src.utils.load_dataset_math import load_gsm8k
        _assert_contract(load_gsm8k())

    def test_svamp(self):
        from src.utils.load_dataset_math import load_svamp
        _assert_contract(load_svamp())

    def test_sst2(self):
        from src.utils.load_dataset_classification import (
            load_sst2,
        )
        _assert_contract(load_sst2())

    def test_rucola(self):
        from src.utils.load_dataset_russian import load_rucola
        _assert_contract(load_rucola())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests**

Run: `uv run python -m unittest test.utils.test_new_loaders -v`
Expected: all PASS (requires network on first run to populate HF cache).

- [ ] **Step 3: Commit**

```bash
git add test/utils/test_new_loaders.py
git commit -m "test(bench): add loader contract smoke tests"
```

---

## Task 8: Benchmark config + CLI runner

**Files:**
- Create: `src/solutions/PE2/extra_benchmarks_config.py`
- Create: `src/solutions/PE2/extra_benchmarks_test.py`

Wires every loader into `BenchmarkTask`s with the right `task`, `metric`, start prompt, and `problem_description`, then runs all four methods (`pe2`, `pe2_sgr`, `ape`, `opro`) via `ParallelBenchmarkRunner` using a model from the ladder.

- [ ] **Step 1: Write the config module**

Create `src/solutions/PE2/extra_benchmarks_config.py`:

```python
"""Benchmark configs for the new thesis datasets.

Groups: ifeval, math (gsm8k+svamp), classification
(sst2/agnews/trec/subj), russian (rucola).
"""

from src.utils.load_dataset_ifeval import load_ifeval
from src.utils.load_dataset_math import load_gsm8k, load_svamp
from src.utils.load_dataset_classification import (
    load_sst2,
    load_agnews,
    load_trec,
    load_subj,
)
from src.utils.load_dataset_russian import load_rucola

_NUMERIC_PROMPT = (
    "Solve the problem. Put only the final numeric answer "
    "inside <ans></ans> tags."
)


def _label_prompt(labels: str) -> str:
    return (
        "Classify the input. Respond with exactly one of "
        f"[{labels}] inside <ans></ans> tags."
    )


# key -> config (loader-based, like sgr_advantage_config)
BENCHMARKS = {
    "ifeval": {
        "start_prompt": (
            "Follow all instructions in the request exactly."
        ),
        "task": "generation",
        "metric": "ifeval",
        "problem_description": (
            "instruction following with verifiable constraints"
        ),
        "loader": lambda: load_ifeval(),
    },
    "gsm8k": {
        "start_prompt": _NUMERIC_PROMPT,
        "task": "generation",
        "metric": "em",
        "problem_description": "grade-school math word problems",
        "loader": lambda: load_gsm8k(),
    },
    "svamp": {
        "start_prompt": _NUMERIC_PROMPT,
        "task": "generation",
        "metric": "em",
        "problem_description": "arithmetic word problems",
        "loader": lambda: load_svamp(),
    },
    "sst2": {
        "start_prompt": _label_prompt("positive, negative"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "movie review sentiment",
        "loader": lambda: load_sst2(),
    },
    "agnews": {
        "start_prompt": _label_prompt(
            "world, sports, business, sci/tech"
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "news topic classification",
        "loader": lambda: load_agnews(),
    },
    "trec": {
        "start_prompt": _label_prompt(
            "abbreviation, entity, description, human, "
            "location, number"
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "question type classification",
        "loader": lambda: load_trec(),
    },
    "subj": {
        "start_prompt": _label_prompt("objective, subjective"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "subjectivity classification",
        "loader": lambda: load_subj(),
    },
    "rucola": {
        "start_prompt": _label_prompt("acceptable, unacceptable"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "Russian linguistic acceptability"
        ),
        "loader": lambda: load_rucola(),
    },
}
```

- [ ] **Step 2: Write the CLI runner**

Create `src/solutions/PE2/extra_benchmarks_test.py`:

```python
"""Run the new thesis benchmarks across the four methods.

Usage:
    uv run python src/solutions/PE2/extra_benchmarks_test.py \\
        --benchmark gsm8k --model mid --method pe2_sgr \\
        --sample 50

Run all methods on one benchmark by omitting --method.
"""

import argparse
import json
import sys
from pathlib import Path

project_path = str(
    Path(__file__).resolve().parent.parent.parent.parent
)
sys.path.append(project_path)

from src.solutions.PE2.extra_benchmarks_config import (  # noqa: E402
    BENCHMARKS,
)
from src.solutions.PE2.local_models import make_llm  # noqa: E402
from src.utils.parallel_runner import (  # noqa: E402
    BenchmarkTask,
    ParallelBenchmarkRunner,
)

METHODS = ["pe2", "pe2_sgr", "ape", "opro"]


def _sample_fn(sample_size):
    import random

    def fn(df):
        if sample_size is None or len(df) <= sample_size:
            return df
        rng = random.Random(42)
        idx = rng.sample(range(len(df)), sample_size)
        return df.iloc[idx]

    return fn


def build_tasks(benchmark, methods, train_steps):
    cfg = BENCHMARKS[benchmark]
    tasks = []
    for method in methods:
        tasks.append(
            BenchmarkTask(
                key=f"{benchmark}/{method}",
                start_prompt=cfg["start_prompt"],
                task=cfg["task"],
                metric=cfg["metric"],
                problem_description=cfg["problem_description"],
                loader=cfg["loader"],
                train_loader=None,
                run_kwargs={
                    "method": method,
                    "verbose": 2,
                    "train_steps": train_steps,
                },
            )
        )
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", required=True, choices=BENCHMARKS.keys()
    )
    parser.add_argument("--model", default="mid")
    parser.add_argument(
        "--method", default=None, choices=METHODS
    )
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        default="logs/extra_benchmarks_results.json",
    )
    args = parser.parse_args()

    methods = [args.method] if args.method else METHODS
    llm = make_llm(args.model)
    tasks = build_tasks(
        args.benchmark, methods, args.train_steps
    )

    runner = ParallelBenchmarkRunner(
        llm=llm,
        max_workers=args.workers,
        sample_fn=_sample_fn(args.sample),
        output_path=Path(args.out),
    )
    runner.run_all(tasks)
    print(f"Done. Results -> {args.out}")


if __name__ == "__main__":
    main()
```

NOTE: confirm `ParallelBenchmarkRunner.__init__` parameter names against `src/utils/parallel_runner.py:48` before running (the repo uses `llm`, `max_workers`, a sample function, and an output path; match the exact keyword names and adjust this call if they differ).

- [ ] **Step 3: Lint**

Run: `uv run flake8 src/solutions/PE2/extra_benchmarks_config.py src/solutions/PE2/extra_benchmarks_test.py`
Expected: clean.

- [ ] **Step 4: Dry-run a tiny generation benchmark (LM Studio mid model loaded)**

Run:
```bash
uv run python src/solutions/PE2/extra_benchmarks_test.py --benchmark gsm8k --method pe2 --sample 3 --train-steps 1
```
Expected: completes without error; `logs/extra_benchmarks_results.json` contains a `gsm8k/pe2` entry with `start_score` and `final_metric`.

- [ ] **Step 5: Dry-run IFEval and a classification benchmark**

Run:
```bash
uv run python src/solutions/PE2/extra_benchmarks_test.py --benchmark ifeval --method pe2_sgr --sample 3 --train-steps 1
uv run python src/solutions/PE2/extra_benchmarks_test.py --benchmark sst2 --method pe2 --sample 5 --train-steps 1
```
Expected: both complete; results JSON gains the new entries with non-null `final_metric`.

- [ ] **Step 6: Commit**

```bash
git add src/solutions/PE2/extra_benchmarks_config.py src/solutions/PE2/extra_benchmarks_test.py
git commit -m "feat(bench): add config + CLI runner for new benchmarks"
```

---

## Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (add a "Benchmarks" section)

- [ ] **Step 1: Document the new benchmarks and how to run them**

Add to `CLAUDE.md` under a new `## Benchmarks` heading: the dataset list (IFEval, GSM8K, SVAMP, SST-2, AG News, TREC, Subj, RuCoLA), the metric each uses (`ifeval`, `em`, `accuracy`), the LM Studio setup steps (`lms server start`, `lms load <model>`), the `MODEL_LADDER` keys (`weak`/`mid`/`strong`), and the example command:
```bash
uv run python src/solutions/PE2/extra_benchmarks_test.py --benchmark <name> --model mid --sample 50
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document new benchmarks and local model setup"
```

---

## Self-Review Notes

- **Spec coverage:** IFEval → Tasks 2,3. GSM8K+SVAMP → Task 4. Classic classification → Task 5. Russian → Task 6. Local model ladder + cross-family API design slot → Task 1. Runner/config → Task 8. Tests → Tasks 2,3,7. Docs → Task 9. All four requested benchmark families + model-ladder requirement covered.
- **Local-only execution:** every run command targets LM Studio via `make_llm`; no API keys required (Anthropic slot is design-only, gated on `CP_ANTHROPIC_KEY`).
- **Type consistency:** loaders all return `pd.DataFrame[input_data, target]`; `check_instruction(id, response, kwargs)` signature is consistent between `ifeval_checkers.py`, its tests, and `IFEvalMetric._compute_raw`; `make_llm(name)` used identically in Tasks 1 and 8.
- **Known external risks (verify at execution, do not guess):** HF dataset ids drift (`stanfordnlp/sst2`, `fancyzhx/ag_news`, `CogComp/trec`, `SetFit/subj`, `RussianNLP/rucola`, `ChilleD/SVAMP`, `google/IFEval`) — Steps include explicit fallbacks/reporting. `ParallelBenchmarkRunner.__init__` kwarg names must be confirmed against source before Task 8 Step 4.
