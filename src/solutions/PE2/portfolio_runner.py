"""Held-out-validation portfolio runner (leakage-free best-of PE2/SGR).

Splits a benchmark dataset DISJOINTLY into opt / val / test:
  - opt  -> passed to PromptTuner.run (optimizer sees only this)
  - val  -> used to select the portfolio winner (never seen by optimizer)
  - test -> final unbiased evaluation

Usage:
  uv run python src/solutions/PE2/portfolio_runner.py \\
      --benchmark ifeval --seed 42 --n-total 200 --train-steps 3 \\
      --out logs/portfolio_heldout.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from coolprompt.assistant import PromptTuner  # noqa: E402
from coolprompt.evaluator import (  # noqa: E402
    Evaluator,
    validate_and_create_metric,
)
from coolprompt.evaluator.portfolio import select_portfolio  # noqa: E402
from coolprompt.utils.enums import Task  # noqa: E402
from src.solutions.PE2.extra_benchmarks_config import (  # noqa: E402
    BENCHMARKS,
)
from src.solutions.PE2.local_models import make_llm  # noqa: E402

METHODS = ["pe2", "pe2_sgr"]


def _task_enum(task_str: str) -> Task:
    if task_str.lower() == "classification":
        return Task.CLASSIFICATION
    return Task.GENERATION


def _split(inputs, targets, rng, frac_opt, frac_val):
    """Return (opt_in, opt_tgt, val_in, val_tgt, test_in, test_tgt).

    Three disjoint slices determined by frac_opt and frac_val;
    remaining fraction goes to test.  Uses rng for shuffling.
    """
    n = len(inputs)
    indices = list(range(n))
    rng.shuffle(indices)

    n_opt = max(1, int(round(n * frac_opt)))
    n_val = max(1, int(round(n * frac_val)))
    # test gets everything that is left (at least 1 sample)
    n_test = max(1, n - n_opt - n_val)
    # re-cap opt if the three parts exceed n
    if n_opt + n_val + n_test > n:
        n_opt = n - n_val - n_test

    opt_idx = indices[:n_opt]
    val_idx = indices[n_opt:n_opt + n_val]
    test_idx = indices[n_opt + n_val:n_opt + n_val + n_test]

    def _pick(idx):
        ins = [inputs[i] for i in idx]
        tgts = [targets[i] for i in idx]
        return ins, tgts

    return _pick(opt_idx) + _pick(val_idx) + _pick(test_idx)


def run_benchmark(
    benchmark_name,
    llm,
    seed,
    n_total,
    train_steps,
    frac_opt=0.5,
    frac_val=0.25,
):
    cfg = BENCHMARKS[benchmark_name]
    start_prompt = cfg["start_prompt"]
    task_str = cfg["task"]
    metric_str = cfg["metric"]
    problem_description = cfg["problem_description"]
    task_enum = _task_enum(task_str)

    # --- load & sample ---
    raw = cfg["loader"]()
    # loaders return a DataFrame with columns input_data / target
    all_inputs = list(raw["input_data"])
    all_targets = list(raw["target"])
    total = len(all_inputs)
    n = min(n_total, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    inputs = [all_inputs[i] for i in indices]
    targets = [all_targets[i] for i in indices]

    # --- disjoint split ---
    split_rng = random.Random(seed + 1)
    (
        opt_in, opt_tgt,
        val_in, val_tgt,
        test_in, test_tgt,
    ) = _split(inputs, targets, split_rng, frac_opt, frac_val)

    print(
        f"[{benchmark_name}] split: opt={len(opt_in)} "
        f"val={len(val_in)} test={len(test_in)}"
    )

    # --- optimize each method on the opt split only ---
    tuner = PromptTuner(target_model=llm)
    final_prompts = {}
    for method in METHODS:
        print(f"[{benchmark_name}] optimizing {method} ...")
        final_prompts[method] = tuner.run(
            start_prompt,
            task_str,
            opt_in,
            opt_tgt,
            method,
            metric_str,
            problem_description,
            train_steps=train_steps,
        )

    # --- score on val split (portfolio selection) ---
    metric_obj = validate_and_create_metric(task_enum, metric_str, llm)
    evaluator = Evaluator(llm, task_enum, metric_obj)

    val_scores = {}
    for method in METHODS:
        val_scores[method] = evaluator.evaluate(
            final_prompts[method], val_in, val_tgt
        )
        print(
            f"[{benchmark_name}] {method} val_score="
            f"{val_scores[method]:.4f}"
        )

    # --- portfolio selection ---
    results = {m: (final_prompts[m], val_scores[m]) for m in METHODS}
    winner, win_prompt, win_val = select_portfolio(results)
    print(
        f"[{benchmark_name}] portfolio winner={winner} "
        f"val={win_val:.4f}"
    )

    # --- score on test split ---
    portfolio_test = evaluator.evaluate(win_prompt, test_in, test_tgt)
    pe2_test = evaluator.evaluate(
        final_prompts["pe2"], test_in, test_tgt
    )
    sgr_test = evaluator.evaluate(
        final_prompts["pe2_sgr"], test_in, test_tgt
    )

    print(
        f"[{benchmark_name}] test: portfolio={portfolio_test:.4f} "
        f"pe2={pe2_test:.4f} pe2_sgr={sgr_test:.4f}"
    )

    return {
        "winner": winner,
        "val_scores": val_scores,
        "test_score": portfolio_test,
        "pe2_test": pe2_test,
        "pe2_sgr_test": sgr_test,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Held-out-validation portfolio runner"
    )
    parser.add_argument(
        "--benchmark",
        default="all",
        help=(
            "Benchmark name, comma-separated list, or 'all'. "
            f"Available: {', '.join(BENCHMARKS)}"
        ),
    )
    parser.add_argument(
        "--model", default="cross",
        help="Model ladder key (default: cross = gpt-4o-mini)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-total", type=int, default=200,
        dest="n_total",
        help="Total rows to sample per benchmark (default: 200)"
    )
    parser.add_argument(
        "--train-steps", type=int, default=3,
        dest="train_steps",
        help="Optimizer train steps (default: 3)"
    )
    parser.add_argument(
        "--out",
        default="logs/portfolio_heldout.json",
        help="Output JSON path (appends/updates if exists)"
    )
    args = parser.parse_args()

    # resolve benchmarks
    if args.benchmark.lower() == "all":
        bench_names = list(BENCHMARKS.keys())
    else:
        bench_names = [b.strip() for b in args.benchmark.split(",")]
    for b in bench_names:
        if b not in BENCHMARKS:
            parser.error(
                f"Unknown benchmark '{b}'. "
                f"Available: {', '.join(BENCHMARKS)}"
            )

    llm = make_llm(args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        with open(out_path) as fh:
            results = json.load(fh)

    for bench in bench_names:
        key = f"{bench}/portfolio"
        record = run_benchmark(
            bench, llm, args.seed, args.n_total, args.train_steps
        )
        results[key] = record
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[{bench}] saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
