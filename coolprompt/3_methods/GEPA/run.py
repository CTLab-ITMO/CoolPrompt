"""Entry point for running GEPA prompt optimisation over a generic dataset.

Example
-------
::

    export OPENAI_API_KEY=sk-...
    python run.py --dataset gsm8k \\
                  --task_lm openai/gpt-4o-mini \\
                  --reflection_lm openai/gpt-4o-mini \\
                  --max_metric_calls 150 \\
                  --train_size 50 --val_size 100 \\
                  --output outputs/gsm8k
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import (
    DefaultAdapter,
    DefaultDataInst,
    EvaluationResult,
)
from gepa.core.adapter import EvaluationBatch
from gepa.utils.stop_condition import StopperProtocol
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# LangChain helpers
# ---------------------------------------------------------------------------
def _to_lc_messages(messages: list[dict]) -> list:
    """Convert GEPA/OpenAI-style message dicts to LangChain message objects."""
    result = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


class LangChainDefaultAdapter(DefaultAdapter):
    """DefaultAdapter with parallel evaluation via ChatOpenAI.batch().

    DefaultAdapter's built-in callable path calls the model sequentially
    (one HTTP request at a time), which makes large val-set evaluation very
    slow.  This subclass overrides ``evaluate()`` to issue all requests in a
    single ``ChatOpenAI.batch()`` call, which LangChain executes in a thread
    pool while still honouring the shared ``InMemoryRateLimiter``.
    """

    def __init__(
        self,
        chat: ChatOpenAI,
        evaluator,
        max_concurrency: int = 10,
    ) -> None:
        # Pass a dummy callable so the parent does not try to import litellm.
        super().__init__(model=lambda msgs: "", evaluator=evaluator)
        self._chat = chat
        self._max_concurrency = max_concurrency

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        system_content = next(iter(candidate.values()))

        message_batches = [
            _to_lc_messages([
                {"role": "system", "content": system_content},
                {"role": "user",   "content": data["input"]},
            ])
            for data in batch
        ]

        lc_responses = self._chat.batch(
            message_batches,
            config=RunnableConfig(max_concurrency=self._max_concurrency),
            return_exceptions=True,
        )

        outputs, scores, trajectories = [], [], ([] if capture_traces else None)

        for data, lc_resp in zip(batch, lc_responses):
            if isinstance(lc_resp, Exception):
                response = ""
                eval_result = EvaluationResult(
                    score=0.0,
                    feedback=f"LLM call failed: {lc_resp!r}",
                )
            else:
                response = lc_resp.content
                eval_result = self.evaluator(data, response)

            outputs.append({"full_assistant_response": response})
            scores.append(eval_result.score)
            if trajectories is not None:
                trajectories.append({
                    "data": data,
                    "full_assistant_response": response,
                    "feedback": eval_result.feedback,
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )


def make_langchain_reflection_lm(chat: ChatOpenAI):
    """Return a LanguageModel callable ``(str) -> str`` backed by ChatOpenAI.

    GEPA's reflection proposer calls this as: new_text = reflection_lm(prompt).
    """
    def _call(prompt: str) -> str:
        return chat.invoke(prompt).content

    return _call


class TokenBudgetStopper:
    """GEPA StopperProtocol that halts optimisation when a token budget is spent.

    Token counts are read from an ``OpenAICallbackHandler`` that is attached
    to both ChatOpenAI instances (task LM and reflection LM) at construction
    time, so every prompt and completion token is counted automatically.
    """

    def __init__(self, budget: int, tracker: OpenAICallbackHandler) -> None:
        self.budget = budget
        self.tracker = tracker

    def __call__(self, gepa_state) -> bool:
        used = self.tracker.total_tokens
        if used >= self.budget:
            print(
                f"\n[run.py] Token budget exhausted: {used:,} / {self.budget:,} tokens. "
                "Stopping optimisation."
            )
            return True
        return False

# Make ``coolprompt`` importable when running from this directory directly.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[2]  # GEPA/ → 3_methods/ → coolprompt/ → project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coolprompt.utils.load_dataset import load_dataset
from coolprompt.utils.utils import get_dataset_split


# ---------------------------------------------------------------------------
# Dataset / metric defaults
# ---------------------------------------------------------------------------
DATASETS = [
    "squad_v2", "gsm8k", "common_gen", "xsum",
    "tweeteval", "mediqa", "concode",
]

DEFAULT_METRICS: dict[str, str] = {
    "squad_v2":     "bert_score",
    "gsm8k":        "exact_match",
    "common_gen":   "bert_score",
    "xsum":         "bert_score",
    "tweeteval":    "f1_mera",
    "mediqa":       "bert_score",
    "code_to_text": "bert_score",
    "concode":      "codebertscore",
}

# Seed system prompts used as GEPA's starting candidate.
SEED_PROMPTS: dict[str, str] = {
    "squad_v2": (
        "Given a context answer on the question."
    ),
    "gsm8k": (
        "Given a context answer on the question."
    ),
    "common_gen": (
        "Create a short sentence using words in list."
    ),
    "xsum": "Summarize the sentence.",
    "tweeteval": (
        "Provide sentiment classification."
    ),
    "mediqa": (
        "Analyze given medical information and answer the question."
    ),
    "code_to_text": (
        "Write a brief natural-language description (docstring) for the "
        "following code snippet."
    ),
    "concode": (
        "write the code"
    ),
}

# HuggingFace subset required for code_to_text (language name).
CODE_TO_TEXT_SUBSET = "python"


# ---------------------------------------------------------------------------
# Metrics  (mirrors EvoPrompt / PromptBreeder implementations)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ANSWER_CUE_RE = re.compile(
    r"(?:answer|result|total)\s*(?:is|:|=)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_TWEETEVAL_LABELS = ["anger", "joy", "optimism", "sadness"]
_TWEETEVAL_SYNONYMS: dict[str, str] = {
    "anger": "anger", "angry": "anger", "mad": "anger",
    "furious": "anger", "rage": "anger", "enraged": "anger",
    "outraged": "anger", "hostile": "anger",
    "joy": "joy", "joyful": "joy", "happy": "joy",
    "happiness": "joy", "glad": "joy", "excited": "joy",
    "delight": "joy", "delighted": "joy", "elated": "joy",
    "optimism": "optimism", "optimistic": "optimism", "hope": "optimism",
    "hopeful": "optimism", "positive": "optimism",
    "sadness": "sadness", "sad": "sadness", "sorrow": "sadness",
    "unhappy": "sadness", "depressed": "sadness", "grief": "sadness",
    "miserable": "sadness", "gloomy": "sadness",
}


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def _extract_last_number(text: str) -> str | None:
    cleaned = (text or "").replace(",", "")
    matches = _NUM_RE.findall(cleaned)
    return matches[-1] if matches else None


def _extract_number_answer(text: str) -> str | None:
    if not text:
        return None
    cue = _ANSWER_CUE_RE.findall(text)
    if cue:
        return cue[-1].replace(",", "")
    return _extract_last_number(text)


def _canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f.is_integer() else repr(f)


def _extract_tweeteval_label(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    best_label, best_pos = "", len(lowered) + 1
    for label in _TWEETEVAL_LABELS:
        m = re.search(rf"\b{re.escape(label)}\b", lowered)
        if m is not None and m.start() < best_pos:
            best_pos, best_label = m.start(), label
    if best_label:
        return best_label
    for word in words:
        canonical = _TWEETEVAL_SYNONYMS.get(word)
        if canonical:
            return canonical
    return words[0] if words else ""


def extract_answer(dataset: str, text: str) -> str:
    """Dataset-aware answer extraction (mirrors EvoPrompt logic)."""
    if dataset == "gsm8k":
        return _canonical_number(_extract_number_answer(text)) or ""
    if dataset == "tweeteval":
        return _extract_tweeteval_label(text)
    return text


# ---- individual metric functions ----

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def f1_mera(pred: str, gold: str) -> float:
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


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    p = re.findall(r"\w+", pred.lower())
    g = re.findall(r"\w+", gold.lower())
    if not p or not g:
        return 0.0
    lcs = _lcs_length(p, g)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(g)
    return 2 * prec * rec / (prec + rec)


_BERT_SCORE_FN: Any = None  # cached or False if unavailable
_BERT_SCORE_WARNED = False


def _get_bert_score_fn():
    global _BERT_SCORE_FN, _BERT_SCORE_WARNED
    if _BERT_SCORE_FN is not None:
        return _BERT_SCORE_FN or None
    try:
        from bert_score import score as _bs
        _BERT_SCORE_FN = _bs
        return _BERT_SCORE_FN
    except Exception as exc:
        _BERT_SCORE_FN = False
        if not _BERT_SCORE_WARNED:
            _BERT_SCORE_WARNED = True
            print(
                f"[run.py] WARNING: bert-score unavailable ({exc!r}); "
                "falling back to ROUGE-L F1."
            )
        return None


def bert_score_f1(pred: str, gold: str) -> float:
    fn = _get_bert_score_fn()
    if fn is None:
        return rouge_l_f1(pred, gold)
    try:
        _, _, F1 = fn(
            [pred or ""], [gold or ""],
            lang="en", rescale_with_baseline=False, verbose=False,
        )
        return float(F1[0].item())
    except Exception:
        return rouge_l_f1(pred, gold)


_CODE_BERT_SCORER: Any = None  # cached BERTScorer(lang="java") or False


def _get_code_bert_scorer():
    global _CODE_BERT_SCORER
    if _CODE_BERT_SCORER is not None:
        return _CODE_BERT_SCORER or None
    try:
        from code_bert_score import BERTScorer
        _CODE_BERT_SCORER = BERTScorer(lang="java")
        return _CODE_BERT_SCORER
    except Exception as exc:
        _CODE_BERT_SCORER = False
        print(f"[run.py] WARNING: code-bert-score unavailable ({exc!r}); "
              "falling back to ROUGE-L F1 for concode.")
        return None


def codebertscore_f1(pred: str, gold: str) -> float:
    scorer = _get_code_bert_scorer()
    if scorer is None:
        return rouge_l_f1(pred, gold)
    try:
        _, _, F1 = scorer.score(cands=[pred or ""], refs=[gold or ""])
        return float(F1[0])
    except Exception:
        return rouge_l_f1(pred, gold)


METRIC_FUNCS: dict[str, Any] = {
    "bert_score":      bert_score_f1,
    "exact_match":     exact_match,
    "f1_mera":         f1_mera,
    "codebertscore":   codebertscore_f1,
}


# ---------------------------------------------------------------------------
# GEPA Evaluator
# ---------------------------------------------------------------------------
class DatasetEvaluator:
    """GEPA-compatible Evaluator for our 8 datasets.

    Implements the ``Evaluator`` protocol expected by ``DefaultAdapter``:
    ``__call__(data: DefaultDataInst, response: str) -> EvaluationResult``
    """

    def __init__(self, dataset: str, metric: str) -> None:
        self.dataset = dataset
        self.metric_name = metric
        self.metric_fn = METRIC_FUNCS[metric]

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        gold = data["answer"]
        pred_ans = extract_answer(self.dataset, response or "")
        gold_ans = extract_answer(self.dataset, gold or "")
        score = float(self.metric_fn(pred_ans, gold_ans))

        if score >= 1.0:
            feedback = "Correct."
        elif score > 0.5:
            feedback = (
                f"Partially correct ({self.metric_name}={score:.3f}). "
                f"Expected: {gold!r}. Got: {response!r}"
            )
        else:
            feedback = (
                f"Incorrect ({self.metric_name}={score:.3f}). "
                f"Expected: {gold!r}. Got: {response!r}"
            )
        return EvaluationResult(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_examples(
    inputs: list[str], targets: list[str]
) -> list[DefaultDataInst]:
    return [
        {"input": inp, "additional_context": {}, "answer": tgt}
        for inp, tgt in zip(inputs, targets)
    ]


def evaluate_on_test(
    adapter: DefaultAdapter,
    candidate: dict[str, str],
    test_examples: list[DefaultDataInst],
) -> float:
    if not test_examples:
        return 0.0
    eval_batch = adapter.evaluate(test_examples, candidate, capture_traces=False)
    return sum(eval_batch.scores) / len(eval_batch.scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GEPA prompt optimisation over one of the 8 supported datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Budget shortcuts (--population_size / --num_epochs):
  GEPA's unit of work is a single-example evaluation ("metric call").
  These flags offer an EvoPrompt-style parameterisation that maps to GEPA's
  budget as follows:

    reflection_minibatch_size = train_size // population_size
    max_metric_calls = population_size * num_epochs * 2 * reflection_minibatch_size
                     ≈ 2 * num_epochs * train_size

  Example: population_size=5, num_epochs=5, train_size=50
    → reflection_minibatch_size = 10
    → max_metric_calls = 500

  When either flag is set both must be provided; they override --max_metric_calls.
""",
    )
    p.add_argument("--dataset", required=True, choices=DATASETS,
                   help="Dataset to optimise on.")
    p.add_argument("--metric", default=None, choices=list(METRIC_FUNCS),
                   help="Evaluation metric. Defaults to per-dataset default.")
    p.add_argument("--task_lm", default="openai/gpt-4o-mini",
                   help="LiteLLM model string for the task LM (optimised).")
    p.add_argument("--reflection_lm", default=None,
                   help="LiteLLM model string for the reflection LM. "
                        "Defaults to task_lm.")
    p.add_argument("--max_metric_calls", type=int, default=None,
                   help="Optimisation budget (total metric calls). "
                        "Ignored when --population_size/--num_epochs are set. "
                        "Default: 500.")
    p.add_argument("--population_size", type=int, default=None,
                   help="Number of GEPA steps per epoch "
                        "(analogous to EvoPrompt popsize). "
                        "Must be used together with --num_epochs.")
    p.add_argument("--num_epochs", type=int, default=None,
                   help="Number of optimisation epochs "
                        "(analogous to EvoPrompt budget). "
                        "Must be used together with --population_size.")
    p.add_argument("--train_size", type=int, default=50,
                   help="Examples used in GEPA trainset. Default: 50.")
    p.add_argument("--val_size", type=int, default=100,
                   help="Examples used in GEPA valset. Default: 100.")
    p.add_argument("--test_size", type=int, default=100,
                   help="Examples for final test evaluation. Default: 100.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed. Default: 0.")
    p.add_argument("--api_key", default=None,
                   help="API key for the LLM provider. "
                        "Falls back to OPENAI_API_KEY env var if not set.")
    p.add_argument("--base_url", default=None,
                   help="Custom API base URL (e.g. for proxies or local servers).")
    p.add_argument("--max_tokens", type=int, default=4000,
                   help="Max tokens per model response. Default: 10000.")
    p.add_argument("--timeout", type=int, default=60,
                   help="HTTP timeout in seconds. Default: 60.")
    p.add_argument("--max_retries", type=int, default=2,
                   help="Number of HTTP retries on failure. Default: 2.")
    p.add_argument("--requests_per_second", type=float, default=3.0,
                   help="Rate limit: max requests per second. Default: 3.0.")
    p.add_argument("--max_concurrency", type=int, default=50,
                   help="Max parallel ChatOpenAI calls per batch. Default: 50.")
    p.add_argument("--token_budget", type=int, default=None,
                   help="Stop optimisation after this many total tokens are spent "
                        "(prompt + completion, task LM + reflection LM). "
                        "Overrides --max_metric_calls when set.")
    p.add_argument("--output", default=None,
                   help="Output directory. Defaults to outputs/<dataset>.")
    p.add_argument("--results_json", default=None,
                   help="Path for the JSON results file. "
                        "Defaults to <output>/results.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = args.dataset
    metric = args.metric or DEFAULT_METRICS[dataset]
    reflection_lm = args.reflection_lm or args.task_lm
    output_dir = Path(args.output or f"outputs/{dataset}")
    results_json = args.results_json or str(output_dir / "results.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve budget ---------------------------------------------------------
    # When token_budget is active, num_epochs has no effect: max_metric_calls is
    # overridden to None later and only the token stopper controls the run.
    # population_size is still useful alone: it sets reflection_minibatch_size
    # (= train_size // population_size), which controls how many examples GEPA
    # uses per reflection step.
    using_token_budget = args.token_budget is not None

    if using_token_budget:
        # Only population_size matters (minibatch granularity); num_epochs ignored.
        if args.num_epochs is not None:
            print("[run.py] WARNING: --num_epochs is ignored when --token_budget is set.")
        reflection_minibatch_size = (
            max(1, args.train_size // args.population_size)
            if args.population_size is not None
            else None  # GEPA default (3)
        )
        max_metric_calls = None  # controlled entirely by TokenBudgetStopper
        both_set = False
    else:
        both_set = args.population_size is not None and args.num_epochs is not None
        one_set = (args.population_size is None) != (args.num_epochs is None)
        if one_set:
            raise ValueError(
                "--population_size and --num_epochs must be provided together "
                "(or use --token_budget to omit --num_epochs)."
            )
        if both_set:
            reflection_minibatch_size = max(1, args.train_size // args.population_size)
            max_metric_calls = (
                args.population_size * args.num_epochs * 2 * reflection_minibatch_size
            )
        else:
            reflection_minibatch_size = None  # GEPA default (3)
            max_metric_calls = args.max_metric_calls if args.max_metric_calls is not None else 500

    # ---- load & split train data -------------------------------------------
    total_train = args.train_size + args.val_size
    val_fraction = args.val_size / total_train

    subset = CODE_TO_TEXT_SUBSET if dataset == "code_to_text" else None
    train_inputs, train_targets = load_dataset(
        dataset, split="train", subset=subset, size=total_train
    )
    tr_in, val_in, tr_tgt, val_tgt = get_dataset_split(
        train_inputs, train_targets,
        validation_size=val_fraction,
        seed=args.seed,
    )
    trainset = build_examples(list(tr_in), list(tr_tgt))
    valset = build_examples(list(val_in), list(val_tgt))

    # ---- load test data -------------------------------------------------------
    test_inputs, test_targets = load_dataset(
        dataset, split="test", subset=subset, size=args.test_size
    )
    testset = build_examples(test_inputs, test_targets)

    if args.token_budget is not None:
        budget_info = (
            f"token_budget={args.token_budget:,}"
            + (f"  population_size={args.population_size}"
               f" → reflection_minibatch_size={reflection_minibatch_size}"
               if args.population_size is not None else "")
        )
    elif both_set:
        budget_info = (
            f"population_size={args.population_size} num_epochs={args.num_epochs}"
            f" → reflection_minibatch_size={reflection_minibatch_size}"
            f" max_metric_calls={max_metric_calls}"
        )
    else:
        budget_info = f"max_metric_calls={max_metric_calls}"
    print(f"[run.py] dataset={dataset}  metric={metric}")
    print(f"[run.py] train={len(trainset)}  val={len(valset)}  test={len(testset)}")
    print(f"[run.py] task_lm={args.task_lm}  reflection_lm={reflection_lm}")
    print(f"[run.py] budget: {budget_info}  seed={args.seed}")

    # ---- build ChatOpenAI clients -------------------------------------------
    from langchain_core.rate_limiters import InMemoryRateLimiter

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or None

    # Shared rate limiter — both task and reflection LM hit the same endpoint.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=args.requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    # Shared token tracker — attached to both models so every call is counted.
    token_tracker = OpenAICallbackHandler()

    common_kwargs: dict[str, Any] = {
        "max_tokens":  args.max_tokens,
        "timeout":     args.timeout,
        "max_retries": args.max_retries,
        "rate_limiter": rate_limiter,
        "callbacks":   [token_tracker],
    }
    if api_key:
        common_kwargs["api_key"] = api_key
    if base_url:
        common_kwargs["base_url"] = base_url

    reflection_lm_callable = make_langchain_reflection_lm(
        ChatOpenAI(model=reflection_lm, temperature=0.5, **common_kwargs)
    )

    # ---- adapter & seed candidate -------------------------------------------
    evaluator = DatasetEvaluator(dataset=dataset, metric=metric)
    adapter = LangChainDefaultAdapter(
        chat=ChatOpenAI(model=args.task_lm, temperature=0, **common_kwargs),
        evaluator=evaluator,
        max_concurrency=args.max_concurrency,
    )
    seed_candidate = {"system_prompt": SEED_PROMPTS[dataset]}

    # ---- resolve stop conditions --------------------------------------------
    stop_callbacks = []
    if using_token_budget:
        stop_callbacks.append(TokenBudgetStopper(args.token_budget, token_tracker))

    # ---- GEPA optimisation --------------------------------------------------
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm_callable,
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callbacks or None,
        seed=args.seed,
        run_dir=str(output_dir / f"gepa_run_seed{args.seed}"),
        display_progress_bar=True,
    )

    best_idx = result.best_idx
    best_prompt = result.best_candidate.get("system_prompt", "")
    best_val_score = float(result.val_aggregate_scores[best_idx])

    total_tokens_used = token_tracker.total_tokens
    print(f"\n[run.py] Best val score : {best_val_score:.4f}")
    print(f"[run.py] Best prompt    : {best_prompt!r}")
    print(f"[run.py] Tokens used    : {total_tokens_used:,}"
          + (f" / {args.token_budget:,} budget" if args.token_budget else ""))

    # ---- final test evaluation -----------------------------------------------
    print(f"[run.py] Evaluating best prompt on {len(testset)} test examples …")
    test_score = evaluate_on_test(adapter, result.best_candidate, testset)
    print(f"[run.py] Test score: {test_score:.4f}")

    # ---- top-k candidates ---------------------------------------------------
    ranked = sorted(
        enumerate(result.candidates),
        key=lambda x: result.val_aggregate_scores[x[0]],
        reverse=True,
    )
    top_candidates = [
        {
            "prompt": c.get("system_prompt", ""),
            "val_score": float(result.val_aggregate_scores[i]),
        }
        for i, c in ranked[:5]
    ]

    # ---- persist results ----------------------------------------------------
    record: dict[str, Any] = {
        "dataset": dataset,
        "metric": metric,
        "task_lm": args.task_lm,
        "reflection_lm": reflection_lm,
        "population_size": args.population_size,
        "num_epochs": args.num_epochs,
        "reflection_minibatch_size": reflection_minibatch_size,
        "token_budget": args.token_budget,
        "total_tokens_used": total_tokens_used,
        "max_metric_calls": max_metric_calls,
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "seed": args.seed,
        "best_prompt": best_prompt,
        "best_val_score": best_val_score,
        "test_score": test_score,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": len(result.candidates),
        "top_candidates": top_candidates,
    }

    os.makedirs(os.path.dirname(os.path.abspath(results_json)) or ".", exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)

    print(f"[run.py] Results written to {results_json}")


if __name__ == "__main__":
    main()
