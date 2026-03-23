"""Run PE2 paper benchmarks (69 tasks across 4 categories).

Usage examples:
    # Full run with PE2+SGR:
    uv run python pe2_paper_test.py --method pe2_sgr

    # Math only with plain PE2:
    uv run python pe2_paper_test.py --method pe2 --categories math

    # BBH + II:
    uv run python pe2_paper_test.py --method pe2_sgr --categories bbh,ii

    # Specific split for II/CF tasks:
    uv run python pe2_paper_test.py --method pe2_sgr --ii_cf_split 2

    # Parallel run (2 workers):
    uv run python pe2_paper_test.py --method ape --workers 2
"""

import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI

project_path = str(
    Path(__file__).resolve().parent.parent.parent.parent
)
sys.path.append(project_path)

from coolprompt.assistant import PromptTuner  # noqa: E402
from src.solutions.PE2.pe2_paper_config import (  # noqa: E402
    build_pe2_paper_config,
)
from src.utils.parallel_runner import (  # noqa: E402
    BenchmarkTask,
    ParallelBenchmarkRunner,
)

# Category short names → prefixes in config keys
CATEGORY_PREFIXES = {
    "math": "math/",
    "bbh": "bbh/",
    "ii": "ii/",
    "cf": "cf/",
}


def sample(
    data: pd.DataFrame,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly sample rows from a DataFrame."""
    if sample_size is None or len(data) <= sample_size:
        return data
    rng = random.Random(seed)
    indices = rng.sample(range(len(data)), sample_size)
    return data.iloc[indices]


def run_pe2_paper(
    method: str,
    categories: list[str],
    ii_cf_split: int,
    sample_size: int,
    train_steps: int,
    output_path: Path,
    sgr_log_path: Path | None,
    workers: int,
):
    """Run PE2 paper benchmarks and save incrementally.

    Args:
        method: "pe2", "pe2_sgr", "ape", or "opro".
        categories: List of category short names to run.
        ii_cf_split: Split index (0-4) for II/CF tasks.
        sample_size: Max samples per task.
        train_steps: Number of training steps.
        output_path: Path for results JSON.
        sgr_log_path: Path for SGR reasoning log
            (only used when method is "pe2_sgr").
        workers: Number of parallel workers.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("CP_OPENAI_KEY"),
        max_retries=10,
        request_timeout=120,
    )

    config = build_pe2_paper_config(ii_cf_split)

    # Filter to requested categories
    prefixes = [
        CATEGORY_PREFIXES[c] for c in categories
    ]
    tasks_cfg = {
        k: v for k, v in config.items()
        if any(k.startswith(p) for p in prefixes)
    }

    print(
        f"Running {len(tasks_cfg)} tasks "
        f"(categories: {', '.join(categories)}, "
        f"method: {method})"
    )

    # Build run_kwargs shared by every task
    base_kwargs: dict = {
        "method": method,
        "verbose": 2,
        "train_as_test": True,
        "train_steps": train_steps,
    }
    if method == "pe2_sgr" and sgr_log_path:
        base_kwargs["log_path"] = str(sgr_log_path)

    # Build BenchmarkTask list
    bench_tasks = [
        BenchmarkTask(
            key=key,
            start_prompt=cfg["start_prompt"],
            task=cfg["task"],
            metric=cfg["metric"],
            problem_description=cfg["problem_description"],
            loader=cfg["loader"],
            run_kwargs=dict(base_kwargs),
        )
        for key, cfg in tasks_cfg.items()
    ]

    sample_fn = (
        lambda df: sample(df, sample_size)
    )

    runner = ParallelBenchmarkRunner(
        llm=llm,
        output_path=output_path,
        max_workers=workers,
        sample_fn=sample_fn,
    )
    return runner.run_all(bench_tasks)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PE2 paper benchmarks"
    )
    parser.add_argument(
        "--method",
        choices=["pe2", "pe2_sgr", "ape", "opro"],
        default="pe2_sgr",
        help="Optimization method (default: pe2_sgr)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="math,bbh,ii,cf",
        help=(
            "Comma-separated categories to run: "
            "math,bbh,ii,cf (default: all)"
        ),
    )
    parser.add_argument(
        "--ii_cf_split",
        type=int,
        default=0,
        help=(
            "Split index 0-4 for II/CF tasks "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Max samples per task (default: 100)",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=3,
        help="Training steps (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers (default: 1 = sequential; "
            "recommend 2-4 for parallel runs)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    categories = [
        c.strip() for c in args.categories.split(",")
    ]

    for c in categories:
        if c not in CATEGORY_PREFIXES:
            print(
                f"Unknown category: {c}. "
                f"Valid: {list(CATEGORY_PREFIXES.keys())}"
            )
            sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            f"./logs/pe2_paper_{args.method}_results.json"
        )

    sgr_log_path = None
    if args.method == "pe2_sgr":
        sgr_log_path = Path(
            "./logs/pe2_paper_sgr_reasoning.jsonl"
        )

    run_pe2_paper(
        method=args.method,
        categories=categories,
        ii_cf_split=args.ii_cf_split,
        sample_size=args.sample_size,
        train_steps=args.train_steps,
        output_path=output_path,
        sgr_log_path=sgr_log_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
