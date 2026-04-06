"""Ablation scoring: iterate meta-prompt variants × benchmarks, collect metrics.

Features:
  - Checkpoint/resume: saves results after each (variant, benchmark) pair.
    On restart, skips already-completed pairs. Failed pairs (with "error" key)
    are automatically retried.
  - File logging: all output goes to both stdout and a log file.
"""

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnableConfig

project_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
sys.path.insert(0, project_path)

from coolprompt.optimizer.hype.hype import HyPEOptimizer
from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.evaluator.metrics import BaseMetric
from coolprompt.utils.var_validation import validate_task
from coolprompt.utils.enums import Task
from coolprompt.utils.parsing import extract_answer
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)

from src.solutions.HyPE.config_dict import config_dict
from src.utils.load_dataset_coolprompt import tweeteval_emotions


# ── constants ────────────────────────────────────────────────────────────────

TEMPLATE_MAP = {
    "classification": CLASSIFICATION_TASK_TEMPLATE,
    "generation": GENERATION_TASK_TEMPLATE,
}

QUERY_SUFFIX = (
    "\n\n{META_INFO_BLOCK}"
    "User query:\n<user_query>\n{QUERY}\n</user_query>\n"
)

ANS_TAGS = ("<ans>", "</ans>")


# ── logging setup ────────────────────────────────────────────────────────────

def setup_file_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes to both file and stdout."""
    logger = logging.getLogger("ablation_score")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ── helpers ──────────────────────────────────────────────────────────────────

def sample(
    data: pd.DataFrame,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    if sample_size is None:
        return data

    if set(data["target"].unique()).issubset(set(tweeteval_emotions)):
        min_class_size = data["target"].value_counts().min()
        per_class = min(sample_size // len(tweeteval_emotions), min_class_size)
        balanced_parts = [
            df.sample(per_class, random_state=seed)
            for _, df in data.groupby("target")
        ]
        return pd.concat(balanced_parts).reset_index(drop=True)
    else:
        return data.sample(sample_size, random_state=seed)


def load_meta_prompts(path: str | Path) -> dict[str, str]:
    """Load meta-prompt variants from JSON produced by inference.py."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["prompts"]


def make_full_meta_prompt(meta_prompt_body: str) -> str:
    """Append the query/meta-info template that HyPEOptimizer expects."""
    return meta_prompt_body + QUERY_SUFFIX


def compute_format_compliance(raw_answers: list[str]) -> float:
    """Compute the fraction of raw answers that contain <ans>...</ans> tags."""
    if not raw_answers:
        return 0.0
    compliant = sum(
        1
        for ans in raw_answers
        if ANS_TAGS[0] in ans and ANS_TAGS[1] in ans
    )
    return compliant / len(raw_answers)


def evaluate_with_details(
    evaluator: Evaluator,
    prompt: str,
    dataset: list[str],
    targets: list[str | int],
    template: str,
    n_wrong_samples: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run evaluation and return metric, format compliance, and wrong answer samples.

    Single model.batch() call — no extra LLM calls.

    Returns dict with:
        - metric_value: float
        - format_compliance: float (fraction of answers with <ans> tags)
        - wrong_samples: list of dicts with input, raw_answer, parsed_answer, ground_truth
    """
    if evaluator.task == Task.CLASSIFICATION:
        evaluator.metric.extract_labels(targets)

    full_prompts = [
        evaluator._get_full_prompt(prompt, s, template)
        for s in dataset
    ]
    raw_results = evaluator.model.batch(
        full_prompts,
        config=RunnableConfig(max_concurrency=20),
    )
    raw_answers = [
        a.content if isinstance(a, AIMessage) else str(a)
        for a in raw_results
    ]

    format_compliance = compute_format_compliance(raw_answers)
    metric_value = evaluator.metric.compute(raw_answers, targets, dataset)
    parsed_answers = [evaluator.metric.parse_output(a) for a in raw_answers]

    wrong_indices = []
    for i, (parsed, target) in enumerate(zip(parsed_answers, targets)):
        if str(parsed).strip().lower() != str(target).strip().lower():
            wrong_indices.append(i)

    rng = random.Random(seed)
    if len(wrong_indices) > n_wrong_samples:
        wrong_indices = rng.sample(wrong_indices, n_wrong_samples)

    wrong_samples = [
        {
            "input": dataset[i],
            "raw_answer": raw_answers[i],
            "parsed_answer": str(parsed_answers[i]),
            "ground_truth": str(targets[i]),
        }
        for i in wrong_indices
    ]

    return {
        "metric_value": metric_value,
        "format_compliance": format_compliance,
        "wrong_samples": wrong_samples,
    }


# ── checkpoint I/O ───────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load existing checkpoint or return empty structure."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "results": {}}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save checkpoint (write to tmp then rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.rename(path)


def is_bench_done(results: dict, variant_name: str, bench_name: str) -> bool:
    """Check if a (variant, bench) pair is already completed successfully."""
    variant = results.get(variant_name)
    if variant is None:
        return False
    bench = variant.get("benchmarks", {}).get(bench_name)
    if bench is None:
        return False
    # Retry if there was an error
    if "error" in bench:
        return False
    # Retry if metric_value is None (incomplete)
    if bench.get("metric_value") is None:
        return False
    return True


# ── main scoring loop ────────────────────────────────────────────────────────

def run_ablation_scoring(
    meta_prompts_path: str | Path,
    output_file: Path,
    sample_size: int = 200,
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Score every meta-prompt variant on every benchmark.

    Supports checkpoint/resume: loads existing results from output_file,
    skips completed (variant, benchmark) pairs, retries failed ones.
    Saves after each (variant, benchmark) completion.
    """
    log_path = output_file.with_suffix(".log")
    log = setup_file_logger(log_path)
    log.info(f"=== Ablation scoring started ===")
    log.info(f"Log file: {log_path}")
    log.info(f"Checkpoint file: {output_file}")

    # ── LLM setup ────────────────────────────────────────────────────────
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=15,
        check_every_n_seconds=0.1,
        max_bucket_size=50,
    )
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        max_completion_tokens=4000,
        max_retries=5,
        rate_limiter=rate_limiter,
        api_key="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",
        extra_body={"allowed_providers": ["google-vertex", "azure"]},
        base_url="https://openrouter.ai/api/v1",
    )

    hype_opt = HyPEOptimizer(model=llm)

    # ── load meta-prompt variants ────────────────────────────────────────
    prompts_map = load_meta_prompts(meta_prompts_path)
    variant_names = sorted(prompts_map.keys())
    log.info(f"Loaded {len(variant_names)} meta-prompt variants from {meta_prompts_path}")

    # ── prepare benchmarks ───────────────────────────────────────────────
    benchmarks: dict[str, dict[str, Any]] = {}
    for task_name, cfg in config_dict.items():
        data_val = cfg["data"][cfg["test_name"]]
        preproc_data = cfg["preproc"](data_val)
        data_sample = sample(preproc_data, sample_size=sample_size)
        dataset = list(data_sample["input_data"])
        target = list(data_sample["target"])

        task_type = validate_task(cfg["task"])
        metric = validate_and_create_metric(task_type, cfg["metric"])
        evaluator = Evaluator(llm, task_type, metric)
        template = TEMPLATE_MAP[cfg["task"]]

        benchmarks[task_name] = {
            "dataset": dataset,
            "target": target,
            "evaluator": evaluator,
            "template": template,
            "metric_name": cfg["metric"],
            "start_prompt": cfg["start_prompt"],
            "problem_description": cfg["problem_description"],
        }

    bench_names = list(benchmarks.keys())
    log.info(f"Prepared {len(benchmarks)} benchmarks: {bench_names}")

    # ── load checkpoint ──────────────────────────────────────────────────
    payload = load_checkpoint(output_file)
    results = payload.get("results", {})

    # Update meta
    payload["meta"] = {
        "started": payload.get("meta", {}).get("started", datetime.now().isoformat()),
        "last_updated": datetime.now().isoformat(),
        "model": model_name,
        "sample_size": sample_size,
        "meta_prompts_source": str(meta_prompts_path),
        "num_variants": len(variant_names),
        "benchmarks": bench_names,
    }

    # ── count work ───────────────────────────────────────────────────────
    total = len(variant_names) * len(benchmarks)
    already_done = sum(
        1
        for vn in variant_names
        for bn in bench_names
        if is_bench_done(results, vn, bn)
    )
    remaining = total - already_done
    log.info(f"Total: {total} | Already done: {already_done} | Remaining: {remaining}")

    # ── build flat work list ──────────────────────────────────────────────
    all_pairs = [
        (vn, bn) for vn in variant_names for bn in bench_names
    ]

    # ── scoring loop with tqdm ────────────────────────────────────────────
    pbar = tqdm(
        all_pairs,
        total=total,
        initial=already_done,
        desc="Scoring",
        unit="pair",
        dynamic_ncols=True,
    )

    prev_variant = None
    for variant_name, bench_name in pbar:
        bench = benchmarks[bench_name]

        # Set up meta-prompt when variant changes
        if variant_name != prev_variant:
            meta_prompt_body = prompts_map[variant_name]
            full_meta_prompt = make_full_meta_prompt(meta_prompt_body)
            hype_opt.set_meta_prompt(full_meta_prompt)
            prev_variant = variant_name

            # Ensure variant entry exists
            if variant_name not in results:
                results[variant_name] = {
                    "meta_prompt": meta_prompt_body,
                    "benchmarks": {},
                }
            elif "meta_prompt" not in results[variant_name]:
                results[variant_name]["meta_prompt"] = meta_prompt_body
            if "benchmarks" not in results[variant_name]:
                results[variant_name]["benchmarks"] = {}

        # Skip if already done
        if is_bench_done(results, variant_name, bench_name):
            pbar.set_postfix_str(f"{variant_name} × {bench_name} [cached]")
            continue

        pbar.set_postfix_str(f"{variant_name} × {bench_name}")
        log.info(f"{variant_name} × {bench_name} ...")

        try:
            result_prompt = hype_opt.optimize(
                prompt=bench["start_prompt"],
                meta_info={
                    "task_description": bench["problem_description"],
                    "required_output_format": (
                        "The final answer MUST be wrapped in <ans> and </ans> XML tags."
                    ),
                },
            )

            eval_result = evaluate_with_details(
                evaluator=bench["evaluator"],
                prompt=result_prompt,
                dataset=bench["dataset"],
                targets=bench["target"],
                template=bench["template"],
                n_wrong_samples=3,
            )

            results[variant_name]["benchmarks"][bench_name] = {
                "result_prompt": result_prompt,
                "metric_name": bench["metric_name"],
                "metric_value": eval_result["metric_value"],
                "format_compliance": eval_result["format_compliance"],
                "wrong_samples": eval_result["wrong_samples"],
            }
            fc = eval_result["format_compliance"]
            mv = eval_result["metric_value"]
            pbar.set_postfix_str(
                f"{variant_name} × {bench_name} ✅ {bench['metric_name']}={mv:.4f} fmt={fc:.0%}"
            )
            log.info(f"  ✅ {bench['metric_name']}={mv:.4f}  fmt={fc:.0%}")

        except Exception as e:
            results[variant_name]["benchmarks"][bench_name] = {
                "result_prompt": None,
                "metric_name": bench["metric_name"],
                "metric_value": None,
                "format_compliance": None,
                "wrong_samples": [],
                "error": str(e),
            }
            pbar.set_postfix_str(f"{variant_name} × {bench_name} ❌")
            log.error(f"  ❌ {variant_name} × {bench_name}: {e}")

        # Save checkpoint after each (variant, bench) pair
        payload["results"] = results
        payload["meta"]["last_updated"] = datetime.now().isoformat()
        save_checkpoint(output_file, payload)

    pbar.close()

    log.info("=== Scoring loop finished ===")
    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary table to stdout."""
    bench_names = list(config_dict.keys())
    col_width = 14
    print("\n📊 Summary (metric / format_compliance):")
    print(f"{'Variant':<45} ", end="")
    for bench_name in bench_names:
        print(f"{bench_name:>{col_width}}", end="")
    print()
    print("-" * (45 + col_width * len(bench_names)))

    for variant_name, variant_data in sorted(results.items()):
        print(f"{variant_name:<45} ", end="")
        for bench_name in bench_names:
            bench_result = variant_data.get("benchmarks", {}).get(bench_name, {})
            mv = bench_result.get("metric_value")
            fc = bench_result.get("format_compliance")
            if mv is not None and fc is not None:
                print(f"{mv:.3f}/{fc:.0%}".rjust(col_width), end="")
            elif "error" in bench_result:
                print(f"{'FAIL':>{col_width}}", end="")
            else:
                print(f"{'---':>{col_width}}", end="")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ablation scoring with checkpoint/resume")
    parser.add_argument(
        "--meta-prompts",
        type=str,
        required=True,
        help="Path to meta_prompts JSON from inference.py",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of samples per benchmark (default: 200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON file path (also used as checkpoint). "
            "Default: ablation_prompts/ablation_scores_<timestamp>.json"
        ),
    )
    args = parser.parse_args()

    if args.output:
        out_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("ablation_prompts")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"ablation_scores_{timestamp}.json"

    results = run_ablation_scoring(
        meta_prompts_path=args.meta_prompts,
        output_file=out_file,
        sample_size=args.sample_size,
        model_name=args.model,
    )

    print(f"\n🎉 Scoring complete! Results saved to {out_file}")
    print_summary(results)


if __name__ == "__main__":
    main()
