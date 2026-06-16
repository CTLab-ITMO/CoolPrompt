"""
Utilities for running GEPA prompt optimization on the 8 local datasets located
under ./datasets/<name>/{train,validation,test}_<name>.json.

Every dataset file follows the same schema:
    {"examples": [{"input": "...", "target": "..."}, ...]}

The 8 supported datasets (treated uniformly as a *generation* task and scored
with BERTScore-F1) are:
    squad_v2, gsm8k, common_gen, xsum, tweeteval, mediqa, code_to_text, concode

This module exposes:
- DATASETS              : list of supported dataset names
- DEFAULT_HPARAMS       : the hyper-parameters required by the task
- load_local_dataset    : load + slice a dataset into (train, val, test)
- make_bertscore_metric : factory returning a GEPA-compatible metric using BERTScore-F1
- build_program         : build a `dspy.ChainOfThought` program for generation
- serialize_detailed_results : convert GEPA's `detailed_results` to JSON-safe dict
- run_optimization      : end-to-end pipeline for one dataset (used by main.ipynb)
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import dspy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset registry & task-specific hyper-parameters
# --------------------------------------------------------------------------- #
DATASETS: List[str] = [
    "squad_v2",
    "gsm8k",
    "common_gen",
    "xsum",
    "tweeteval",
    "mediqa",
    "code_to_text",
    "concode",
]

# Supported optimization metrics.
METRICS: List[str] = ["bert_score", "exact_match", "f1"]

# Per-dataset default metric used when the caller does not explicitly pass one.
#   gsm8k     -> exact_match
#   tweeteval -> f1
#   all other -> bert_score
DATASET_DEFAULT_METRIC: Dict[str, str] = {
    "squad_v2": "bert_score",
    "gsm8k": "exact_match",
    "common_gen": "bert_score",
    "xsum": "bert_score",
    "tweeteval": "f1",
    "mediqa": "bert_score",
    "code_to_text": "bert_score",
    "concode": "bert_score",
}


# Required hyper-parameters from the task description.
# - population size      -> GEPA `reflection_minibatch_size = 10`
# - epochs / iterations  -> GEPA `max_full_evals = 5`
# - splits               -> train/val/test = 50/100/300
# - model                -> gpt-5-nano via OpenAI
# - temperature          -> 1.0
DEFAULT_HPARAMS: Dict[str, Any] = {
    "model": "openai/gpt-5-nano",
    "temperature": 1.0,
    "max_tokens": 16000,
    "train_n": 50,
    "val_n": 100,
    "test_n": 300,
    "population_size": 10,        # reflection_minibatch_size
    "num_iterations": 5,          # max_full_evals
    "num_threads": 8,
    "candidate_selection_strategy": "pareto",
    "use_merge": True,
    "max_merge_invocations": 5,
    "skip_perfect_score": True,
    "perfect_score": 1.0,
    "failure_score": 0.0,
    "seed": 0,
    # When None, the metric is auto-resolved per dataset via DATASET_DEFAULT_METRIC.
    # (gsm8k -> exact_match, tweeteval -> f1, all others -> bert_score)
    "metric": None,
}


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def _read_split(name: str, split: str, root: str | Path) -> List[Dict[str, str]]:
    path = Path(root) / name / f"{split}_{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["examples"]


def _to_examples(rows: List[Dict[str, str]]) -> List[dspy.Example]:
    return [
        dspy.Example(input=str(r["input"]), target=str(r["target"])).with_inputs("input")
        for r in rows
    ]


def load_local_dataset(
    name: str,
    train_n: int = 50,
    val_n: int = 100,
    test_n: int = 300,
    root: str | Path = "datasets",
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Load a local dataset and slice it to (train_n, val_n, test_n).

    Each example has a single input field `input` and a gold field `target`.
    If a split has fewer rows than requested, the available rows are used.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {DATASETS}")

    train = _to_examples(_read_split(name, "train", root))[:train_n]
    val = _to_examples(_read_split(name, "validation", root))[:val_n]
    test = _to_examples(_read_split(name, "test", root))[:test_n]
    return train, val, test


# --------------------------------------------------------------------------- #
# BERTScore metric (generation)
# --------------------------------------------------------------------------- #
def make_bertscore_metric(
    lang: str = "en",
    model_type: str | None = None,
    success_threshold: float = 0.88,
) -> Callable:
    """Return a GEPA-compatible metric using BERTScore-F1.

    The metric returns `dspy.Prediction(score=f1, feedback=str)`. The feedback
    text always contains the gold target so GEPA's reflection LM can learn
    from it.
    """
    from bert_score import score as compute_bertscore  # local import: heavy

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        # The model output field is called `output` (see GenerateAnswer signature).
        pred_text = getattr(prediction, "output", None)
        if pred_text is None:
            # Defensive: support alternative field names.
            pred_text = getattr(prediction, "answer", "") or ""
        pred_text = str(pred_text) if pred_text is not None else ""
        gold_text = str(example["target"])

        # BERTScore requires non-empty strings.
        if not pred_text.strip():
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"Your output was empty. The gold target was: '{gold_text}'. "
                    f"Produce a non-empty answer that semantically matches the target."
                ),
            )

        try:
            _, _, f1 = compute_bertscore(
                [pred_text],
                [gold_text],
                lang=lang,
                model_type=model_type,
                verbose=False,
            )
            f1_score = float(f1.item())
        except Exception as exc:  # pragma: no cover
            return dspy.Prediction(
                score=0.0,
                feedback=f"BERTScore computation failed: {exc}. Gold target was: '{gold_text}'.",
            )

        if f1_score >= success_threshold:
            feedback = (
                f"Great! BERTScore F1 = {f1_score:.3f} (>= {success_threshold}). "
                f"Your output is semantically close to the target. Gold target: '{gold_text}'."
            )
        else:
            feedback = (
                f"BERTScore F1 = {f1_score:.3f} (< {success_threshold}). "
                f"Your output diverged from the gold target. "
                f"Gold target: '{gold_text}'. "
                f"Your output: '{pred_text}'. "
                f"Identify what was missing, wrong, or extra and adjust the instruction "
                f"so future answers match the gold target's style, length and content."
            )
        return dspy.Prediction(score=f1_score, feedback=feedback)

    return metric


# --------------------------------------------------------------------------- #
# Exact-match and token-F1 metrics (SQuAD-style normalization)
# --------------------------------------------------------------------------- #
def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace (SQuAD-style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _extract_pred_text(prediction) -> str:
    pred_text = getattr(prediction, "output", None)
    if pred_text is None:
        pred_text = getattr(prediction, "answer", "") or ""
    return str(pred_text) if pred_text is not None else ""


def make_exact_match_metric() -> Callable:
    """Return a GEPA-compatible exact-match metric (after SQuAD-style normalization)."""

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        pred_text = _extract_pred_text(prediction)
        gold_text = str(example["target"])

        if not pred_text.strip():
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"Your output was empty. The gold target was: '{gold_text}'. "
                    f"Produce an answer that exactly matches the target."
                ),
            )

        norm_pred = _normalize_text(pred_text)
        norm_gold = _normalize_text(gold_text)
        score = 1.0 if norm_pred == norm_gold else 0.0

        if score == 1.0:
            feedback = (
                f"Exact match (score=1.0). Gold target: '{gold_text}'. "
                f"Your output matched the target after normalization."
            )
        else:
            feedback = (
                f"No exact match (score=0.0). "
                f"Gold target: '{gold_text}'. Your output: '{pred_text}'. "
                f"Produce an answer that exactly matches the gold target (case, "
                f"punctuation and articles are ignored, but every other token must match)."
            )
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def make_f1_metric(success_threshold: float = 0.7) -> Callable:
    """Return a GEPA-compatible token-level F1 metric (SQuAD-style)."""

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        pred_text = _extract_pred_text(prediction)
        gold_text = str(example["target"])

        if not pred_text.strip():
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"Your output was empty. The gold target was: '{gold_text}'. "
                    f"Produce a non-empty answer whose tokens match the gold target."
                ),
            )

        f1 = _token_f1(pred_text, gold_text)
        if f1 >= success_threshold:
            feedback = (
                f"Good. Token-level F1 = {f1:.3f} (>= {success_threshold}). "
                f"Gold target: '{gold_text}'."
            )
        else:
            feedback = (
                f"Token-level F1 = {f1:.3f} (< {success_threshold}). "
                f"Gold target: '{gold_text}'. Your output: '{pred_text}'. "
                f"Identify missing or extra tokens and adjust the instruction so "
                f"future answers share more tokens with the gold target."
            )
        return dspy.Prediction(score=f1, feedback=feedback)

    return metric


# --------------------------------------------------------------------------- #
# Metric dispatcher
# --------------------------------------------------------------------------- #
def make_metric(name: str) -> Callable:
    """Return a GEPA-compatible metric by name.

    Supported names: 'bert_score', 'exact_match', 'f1'.
    """
    if name == "bert_score":
        return make_bertscore_metric()
    if name == "exact_match":
        return make_exact_match_metric()
    if name == "f1":
        return make_f1_metric()
    raise ValueError(f"Unknown metric '{name}'. Choose one of: {METRICS}")


def resolve_metric_name(dataset_name: str, metric_name: str | None) -> str:
    """Resolve the metric name to use for `dataset_name`.

    If `metric_name` is given, it is validated and returned.
    Otherwise, the per-dataset default from `DATASET_DEFAULT_METRIC` is used.
    """
    if metric_name is None:
        if dataset_name not in DATASET_DEFAULT_METRIC:
            raise ValueError(
                f"No default metric configured for dataset '{dataset_name}'."
            )
        return DATASET_DEFAULT_METRIC[dataset_name]
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric '{metric_name}'. Choose one of: {METRICS}")
    return metric_name


# --------------------------------------------------------------------------- #
# Program (DSPy signature for generation)
# --------------------------------------------------------------------------- #
class GenerateAnswer(dspy.Signature):
    """Given an input, produce the correct target output for the task."""

    input = dspy.InputField(desc="Task input text.")
    output = dspy.OutputField(desc="The generated answer / target text.")


def build_program() -> dspy.Module:
    """Return a fresh `dspy.ChainOfThought(GenerateAnswer)` program."""
    return dspy.ChainOfThought(GenerateAnswer)


# --------------------------------------------------------------------------- #
# Result serialization
# --------------------------------------------------------------------------- #
def _safe(obj: Any) -> Any:
    """Best-effort recursive conversion of arbitrary objects to JSON-safe data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(v) for v in obj]
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _safe(fn())
            except Exception:
                pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def serialize_detailed_results(detailed_results: Any) -> Dict[str, Any]:
    """Convert GEPA's `detailed_results` to a JSON-safe dict, including a
    per-iteration trace of candidate instructions and their scores."""
    if detailed_results is None:
        return {}

    out: Dict[str, Any] = {}
    for attr in (
        "val_aggregate_scores",
        "val_subscores",
        "best_outputs_valset",
        "best_idx",
        "candidate_scores",
        "candidate_instructions",
        "candidates",
        "history",
        "best_candidate",
    ):
        if hasattr(detailed_results, attr):
            try:
                out[attr] = _safe(getattr(detailed_results, attr))
            except Exception as exc:
                out[attr] = f"<unserializable: {exc}>"

    # Per-iteration trace.
    try:
        candidates = getattr(detailed_results, "candidates", None) or []
        scores = getattr(detailed_results, "val_aggregate_scores", None) or []
        trace = []
        for i, cand in enumerate(candidates):
            entry: Dict[str, Any] = {"iteration": i}
            if isinstance(cand, dict):
                entry["instructions"] = _safe(cand)
            else:
                entry["instructions"] = _safe(getattr(cand, "instructions", cand))
            if i < len(scores):
                entry["val_aggregate_score"] = _safe(scores[i])
            trace.append(entry)
        if trace:
            out["per_iteration_trace"] = trace
    except Exception as exc:  # pragma: no cover
        out["per_iteration_trace_error"] = str(exc)

    return out


# --------------------------------------------------------------------------- #
# End-to-end runner (used by the notebook)
# --------------------------------------------------------------------------- #
def configure_lms(
    openai_api_key: str,
    model: str = "openai/gpt-5-nano",
    temperature: float = 1.0,
    max_tokens: int = 16000,
) -> Tuple[dspy.LM, dspy.LM, Any]:
    """Configure DSPy with the OpenAI model and also return a langchain
    `ChatOpenAI` instance (kept for parity with the task description).

    Returns:
        (task_lm, reflection_lm, chat_openai)
    """
    # Make the API key visible to every downstream client.
    os.environ["OPENAI_API_KEY"] = openai_api_key

    task_lm = dspy.LM(
        model,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_api_key,
        max_tokens=max_tokens
    )
    reflection_lm = dspy.LM(
        model,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_api_key,
        max_tokens=max_tokens
    )
    dspy.configure(lm=task_lm)

    # Build a ChatOpenAI for users who want to call the same model through
    # LangChain. It is not used by GEPA itself (GEPA requires a `dspy.LM`),
    # but it is returned so callers can reuse it for ad-hoc inference.
    chat_openai = None
    try:
        from langchain_openai import ChatOpenAI

        # Strip the "openai/" provider prefix expected by litellm/dspy.
        # lc_model = model.split("/", 1)[1] if "/" in model else model
        chat_openai = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
            max_tokens=max_tokens
        )
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not instantiate langchain_openai.ChatOpenAI: {exc}")

    return task_lm, reflection_lm, chat_openai


def run_optimization(
    dataset_name: str,
    openai_api_key: str,
    *,
    model: str = "openai/gpt-5-nano",
    temperature: float = 1.0,
    max_tokens: int = 16000,
    train_n: int = 50,
    val_n: int = 100,
    test_n: int = 300,
    population_size: int = 10,      # GEPA: reflection_minibatch_size
    num_iterations: int = 5,        # GEPA: max_full_evals
    num_threads: int = 8,
    candidate_selection_strategy: str = "pareto",
    use_merge: bool = True,
    max_merge_invocations: int = 5,
    skip_perfect_score: bool = True,
    perfect_score: float = 1.0,
    failure_score: float = 0.0,
    seed: int = 0,
    metric_name: str | None = None,
    results_dir: str | Path = "results",
    log_root: str | Path = "runs",
    evaluate_on_test: bool = False,
) -> Dict[str, Any]:
    """Run GEPA optimization end-to-end on a single dataset and persist a
    JSON artifact containing the initial prompt, the optimization trace, and
    the final prompt.

    Returns:
        The dict that was written to `<results_dir>/<dataset_name>.json`.
    """
    from dspy.teleprompt import GEPA

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {DATASETS}")

    # 1. Configure LMs (task + reflection) and a ChatOpenAI handle.
    task_lm, reflection_lm, _chat = configure_lms(
        openai_api_key=openai_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 2. Load data.
    train_set, val_set, test_set = load_local_dataset(
        dataset_name, train_n=train_n, val_n=val_n, test_n=test_n
    )

    # 3. Build program and capture the initial prompt.
    program = build_program()
    initial_prompt = program.predict.signature.instructions
    initial_state = program.dump_state()

    # 4. Configure GEPA.
    # Resolve which metric to optimize against (per-dataset default if not given).
    resolved_metric_name = resolve_metric_name(dataset_name, metric_name)

    log_dir = Path(log_root) / dataset_name / resolved_metric_name
    log_dir.mkdir(parents=True, exist_ok=True)

    metric = make_metric(resolved_metric_name)

    optimizer = GEPA(
        metric=metric,
        max_full_evals=num_iterations,                      # "epochs"
        reflection_minibatch_size=population_size,          # "population size"
        reflection_lm=reflection_lm,
        candidate_selection_strategy=candidate_selection_strategy,
        skip_perfect_score=skip_perfect_score,
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        perfect_score=perfect_score,
        failure_score=failure_score,
        num_threads=num_threads,
        track_stats=True,
        track_best_outputs=True,
        seed=seed,
        log_dir=str(log_dir),
        gepa_kwargs={"use_cloudpickle": True},
    )

    # 5. Optimize.
    try:
        optimized_program = optimizer.compile(program, trainset=train_set, valset=val_set)
    except Exception as exc:
        message = str(exc)
        if "Can't pickle" in message and "StringSignature" in message:
            logger.warning(
                "GEPA state serialization failed for dynamically generated DSPy signatures; "
                "retrying optimization with state persistence disabled. Original error: %s",
                exc,
            )
            optimizer = GEPA(
                metric=metric,
                max_full_evals=num_iterations,
                reflection_minibatch_size=population_size,
                reflection_lm=reflection_lm,
                candidate_selection_strategy=candidate_selection_strategy,
                skip_perfect_score=skip_perfect_score,
                use_merge=use_merge,
                max_merge_invocations=max_merge_invocations,
                perfect_score=perfect_score,
                failure_score=failure_score,
                num_threads=num_threads,
                track_stats=True,
                track_best_outputs=True,
                seed=seed,
                log_dir=None,
            )
            optimized_program = optimizer.compile(program, trainset=train_set, valset=val_set)
        else:
            raise

    final_prompt = optimized_program.predict.signature.instructions
    detailed = getattr(optimized_program, "detailed_results", None)

    # 6. (Optional) test-set evaluation with the optimized program.
    test_score: float | None = None
    if evaluate_on_test:
        evaluator = dspy.Evaluate(
            devset=test_set,
            metric=lambda ex, pr, trace=None: float(metric(ex, pr).score),
            num_threads=num_threads,
            display_progress=True,
            display_table=False,
        )
        try:
            test_score = float(evaluator(optimized_program))
        except Exception as exc:  # pragma: no cover
            test_score = None
            print(f"[{dataset_name}] test evaluation failed: {exc}")

    # 7. Assemble + save the artifact.
    artifact: Dict[str, Any] = {
        "dataset": dataset_name,
        "metric": resolved_metric_name,
        "model": model,
        "temperature": temperature,
        "task": "generation",
        "splits": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "hyperparameters": {
            "population_size": population_size,
            "num_iterations": num_iterations,
            "reflection_minibatch_size": population_size,
            "max_full_evals": num_iterations,
            "candidate_selection_strategy": candidate_selection_strategy,
            "skip_perfect_score": skip_perfect_score,
            "use_merge": use_merge,
            "max_merge_invocations": max_merge_invocations,
            "perfect_score": perfect_score,
            "failure_score": failure_score,
            "num_threads": num_threads,
            "seed": seed,
            "metric": resolved_metric_name,
        },
        "initial_prompt": initial_prompt,
        "initial_program_state": _safe(initial_state),
        "optimization_trace": serialize_detailed_results(detailed),
        "final_prompt": final_prompt,
        "final_score": test_score,
        "test_score": test_score,
        "test_metric": resolved_metric_name,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name}_{resolved_metric_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)

    # Save the optimized DSPy program too (handy for inference reuse).
    try:
        optimized_program.save(
            str(results_dir / f"{dataset_name}_{resolved_metric_name}_program.json")
        )
    except Exception as exc:  # pragma: no cover
        print(f"[{dataset_name}] could not save optimized program: {exc}")

    print(f"[{dataset_name}] artifact written to {out_path}")
    return artifact
