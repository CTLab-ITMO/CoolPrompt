"""SGR advantage benchmarks: find tasks where PE2+SGR
outperforms plain PE2.

Phase 1: High-baseline PE2 paper tasks (start > 0.85,
    PE2 delta < 0.02) — SGR's structured refinement
    may help where PE2 plateaus.

Phase 2: Custom multi-constraint generation tasks with
    heterogeneous errors — SGR's error categorization
    adds value when failures have different root causes.

Usage:
    # Phase 1 only (high-baseline PE2 tasks):
    uv run python sgr_advantage_test.py --phase 1

    # Phase 2 only (custom multi-constraint tasks):
    uv run python sgr_advantage_test.py --phase 2

    # Both phases:
    uv run python sgr_advantage_test.py --phase 1,2

    # Specific method:
    uv run python sgr_advantage_test.py --phase 2 \\
        --method pe2

    # Parallel run:
    uv run python sgr_advantage_test.py --phase 1,2 \\
        --workers 2
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

from src.solutions.PE2.sgr_advantage_config import (  # noqa: E402
    build_phase1_config,
    build_phase2_config,
)
from src.utils.parallel_runner import (  # noqa: E402
    BenchmarkTask,
    ParallelBenchmarkRunner,
)


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


def _build_tasks(
    config: dict,
    method: str,
    train_steps: int,
    sgr_log_path: Path | None,
) -> list[BenchmarkTask]:
    """Convert config dict to BenchmarkTask list."""
    base_kwargs: dict = {
        "method": method,
        "verbose": 2,
        "train_steps": train_steps,
    }
    if method == "pe2_sgr" and sgr_log_path:
        base_kwargs["log_path"] = str(sgr_log_path)

    tasks = []
    for key, cfg in config.items():
        run_kwargs = dict(base_kwargs)

        # Phase 2 tasks pass LLM-as-judge params
        for laj_key in (
            "llm_as_judge_criteria",
            "llm_as_judge_custom_templates",
            "llm_as_judge_metric_ceil",
        ):
            if laj_key in cfg:
                run_kwargs[laj_key] = cfg[laj_key]

        tasks.append(
            BenchmarkTask(
                key=key,
                start_prompt=cfg["start_prompt"],
                task=cfg["task"],
                metric=cfg["metric"],
                problem_description=(
                    cfg["problem_description"]
                ),
                loader=cfg["loader"],
                train_loader=cfg.get("train_loader"),
                run_kwargs=run_kwargs,
            )
        )
    return tasks


def run_sgr_advantage(
    phases: list[int],
    method: str,
    ii_cf_split: int,
    sample_size: int,
    train_steps: int,
    output_path: Path,
    sgr_log_path: Path | None,
    workers: int,
):
    """Run SGR advantage benchmarks.

    Args:
        phases: List of phases to run (1, 2, or both).
        method: "pe2", "pe2_sgr", "ape", or "opro".
        ii_cf_split: Split index for II/CF tasks.
        sample_size: Max samples per task.
        train_steps: Number of training steps.
        output_path: Path for results JSON.
        sgr_log_path: Path for SGR reasoning log.
        workers: Number of parallel workers.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("CP_OPENAI_KEY"),
        max_retries=10,
        request_timeout=120,
    )

    all_tasks = []

    if 1 in phases:
        phase1_cfg = build_phase1_config(ii_cf_split)
        all_tasks.extend(
            _build_tasks(
                phase1_cfg, method,
                train_steps, sgr_log_path,
            )
        )
        print(
            f"Phase 1: {len(phase1_cfg)} high-baseline "
            f"tasks"
        )

    if 2 in phases:
        phase2_cfg = build_phase2_config()
        all_tasks.extend(
            _build_tasks(
                phase2_cfg, method,
                train_steps, sgr_log_path,
            )
        )
        print(
            f"Phase 2: {len(phase2_cfg)} custom "
            f"multi-constraint tasks"
        )

    print(
        f"\nTotal: {len(all_tasks)} tasks "
        f"(method: {method})"
    )

    sample_fn = lambda df: sample(df, sample_size)

    runner = ParallelBenchmarkRunner(
        llm=llm,
        output_path=output_path,
        max_workers=workers,
        sample_fn=sample_fn,
    )
    return runner.run_all(all_tasks)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGR advantage benchmarks"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1,2",
        help=(
            "Comma-separated phases to run: "
            "1 (high-baseline), 2 (custom). "
            "Default: 1,2"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["pe2", "pe2_sgr", "ape", "opro"],
        default="pe2_sgr",
        help="Optimization method (default: pe2_sgr)",
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
        help=(
            "Output JSON path (auto-generated if omitted)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers (default: 1 = sequential)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phases = [
        int(p.strip())
        for p in args.phase.split(",")
    ]

    for p in phases:
        if p not in (1, 2):
            print(f"Unknown phase: {p}. Valid: 1, 2")
            sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        phase_str = "_".join(str(p) for p in phases)
        output_path = Path(
            f"./logs/sgr_advantage_{args.method}"
            f"_phase{phase_str}_results.json"
        )

    sgr_log_path = None
    if args.method == "pe2_sgr":
        sgr_log_path = Path(
            "./logs/sgr_advantage_reasoning.jsonl"
        )

    run_sgr_advantage(
        phases=phases,
        method=args.method,
        ii_cf_split=args.ii_cf_split,
        sample_size=args.sample_size,
        train_steps=args.train_steps,
        output_path=output_path,
        sgr_log_path=sgr_log_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
