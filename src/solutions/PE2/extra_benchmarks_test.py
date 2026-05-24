"""Run the new thesis benchmarks across the four methods.

Usage:
    uv run python src/solutions/PE2/extra_benchmarks_test.py \\
        --benchmark gsm8k --model mid --method pe2_sgr \\
        --sample 50

Run all methods on one benchmark by omitting --method.
"""

import argparse
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


def build_tasks(benchmarks, methods, train_steps):
    tasks = []
    for benchmark in benchmarks:
        cfg = BENCHMARKS[benchmark]
        for method in methods:
            tasks.append(
                BenchmarkTask(
                    key=f"{benchmark}/{method}",
                    start_prompt=cfg["start_prompt"],
                    task=cfg["task"],
                    metric=cfg["metric"],
                    problem_description=cfg[
                        "problem_description"
                    ],
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
        "--benchmark",
        required=True,
        help=(
            "Benchmark name, a comma-separated list, or 'all' "
            f"({', '.join(BENCHMARKS)})"
        ),
    )
    parser.add_argument("--model", default="mid")
    parser.add_argument(
        "--method", default=None, choices=METHODS
    )
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        default=None,
        choices=["lmstudio", "openrouter"],
        help="LLM backend (default: CP_BACKEND env or lmstudio)",
    )
    parser.add_argument(
        "--out",
        default="logs/extra_benchmarks_results.json",
    )
    args = parser.parse_args()

    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS)
    else:
        benchmarks = [
            b.strip() for b in args.benchmark.split(",")
            if b.strip()
        ]
    unknown = [b for b in benchmarks if b not in BENCHMARKS]
    if unknown:
        parser.error(
            f"unknown benchmark(s): {', '.join(unknown)}"
        )

    methods = [args.method] if args.method else METHODS
    llm = make_llm(args.model, backend=args.backend)
    tasks = build_tasks(benchmarks, methods, args.train_steps)

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
