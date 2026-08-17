from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from coolprompt.method_evaluation.method_evaluation import evaluate_method


@dataclass(frozen=True)
class BenchmarkSpec:
    """Dataset-specific settings required by ``evaluate_method``."""

    name: str
    task: str
    metric: str
    start_prompt: str


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="squad_v2",
        task="generation",
        metric="bertscore",
        start_prompt="Given a context answer on the question.",
    ),
    BenchmarkSpec(
        name="common_gen",
        task="generation",
        metric="multiref_bertscore",
        start_prompt="Create a short sentence using words in list.",
    ),
    BenchmarkSpec(
        name="gsm8k",
        task="generation",
        metric="em",
        start_prompt="Given a context answer on the question.",
    ),
    BenchmarkSpec(
        name="tweeteval",
        task="classification",
        metric="f1",
        start_prompt="Provide sentiment classification.",
    ),
    BenchmarkSpec(
        name="xsum",
        task="generation",
        metric="bertscore",
        start_prompt="Summarize the sentence.",
    ),
)


def build_config(
    spec: BenchmarkSpec,
    dataset_configuration: str,
    method_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the YAML-style configuration consumed by an optimizer."""

    return {
        "dataset": {
            "name": spec.name,
            "configuration": dataset_configuration,
        },
        "method": method_options or {},
        "task": spec.task,
        "metric": spec.metric,
    }


def run_benchmarks(
    *,
    method: str,
    model: ChatOpenAI,
    output_dir: Path,
    dataset_configuration: str,
    method_options: dict[str, Any] | None = None,
    saving_model_answers: bool = False,
    datasets: Sequence[str] | None = None,
) -> Path:
    """Evaluate one method on selected datasets and return the summary path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    available_datasets = {spec.name for spec in BENCHMARKS}
    selected_datasets = set(datasets) if datasets is not None else available_datasets
    unknown_datasets = selected_datasets - available_datasets
    if unknown_datasets:
        raise ValueError(
            f"Unknown datasets: {sorted(unknown_datasets)}. "
            f"Available: {sorted(available_datasets)}."
        )

    for spec in BENCHMARKS:
        if spec.name not in selected_datasets:
            continue
        output_path = output_dir / f"{method}_{spec.name}.yaml"
        config = build_config(spec, dataset_configuration, method_options)
        if saving_model_answers:
            config["model_answers_output_path"] = str(
                output_dir / f"{method}_{spec.name}_answers.yaml"
            )

        print(f"Running {method} on {spec.name}...")
        evaluate_method(
            method=method,
            model=model,
            config=config,
            start_prompt=spec.start_prompt,
            output_file_path=str(output_path),
            saving_model_answers=saving_model_answers,
        )

    results: list[dict[str, Any]] = []
    for spec in BENCHMARKS:
        output_path = output_dir / f"{method}_{spec.name}.yaml"
        if output_path.exists():
            with output_path.open() as output_file:
                results.append(yaml.safe_load(output_file))

    summary_path = output_dir / f"{method}_summary.yaml"
    with summary_path.open("w") as summary_file:
        yaml.safe_dump(
            {"method": method, "results": results},
            summary_file,
            sort_keys=False,
        )
    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="hyper_light",
        help="Autoprompting method to evaluate (default: hyper_light).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Optional OpenAI-compatible API base URL, for example OpenRouter."
    )
    parser.add_argument(
        "--api-key",
        help="API key. Defaults to the OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per model response (default: unlimited).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=10,
        help="Maximum model requests per second (default: 10).",
    )
    parser.add_argument(
        "--rate-limit-check-seconds",
        type=float,
        default=0.1,
        help="Rate-limiter polling interval in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--rate-limit-bucket-size",
        type=int,
        default=10,
        help="Rate-limiter maximum bucket size (default: 10).",
    )
    parser.add_argument(
        "--dataset-configuration",
        default="-/-/300",
        help="Dataset sizes in train/validation/test form (default: -/-/300).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        help="Override iterative method iteration count.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        help="Override iterative method train batch size.",
    )
    parser.add_argument(
        "--train-pool-size",
        type=int,
        help="Override iterative method train pool size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override iterative method sampling seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("method_evaluation_outputs"),
        help="Directory for per-dataset and summary YAML results.",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset names to run (default: all five).",
    )
    parser.add_argument(
        "--save-model-answers",
        action="store_true",
        help="Save test-set model outputs alongside benchmark results.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Create the requested model and run all five benchmarks."""

    args = parse_args(argv)
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=args.requests_per_second,
        check_every_n_seconds=args.rate_limit_check_seconds,
        max_bucket_size=args.rate_limit_bucket_size,
    )
    model = ChatOpenAI(
        model=args.model,
        api_key="",
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        rate_limiter=rate_limiter,
    )
    method_options = {
        key: value
        for key, value in {
            "n_iterations": args.n_iterations,
            "train_batch_size": args.train_batch_size,
            "train_pool_size": args.train_pool_size,
            "random_seed": args.seed,
        }.items()
        if value is not None
    }
    datasets = (
        [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
        if args.datasets
        else None
    )
    summary_path = run_benchmarks(
        method=args.method,
        model=model,
        output_dir=args.output_dir,
        dataset_configuration=args.dataset_configuration,
        method_options=method_options,
        saving_model_answers=args.save_model_answers,
        datasets=datasets,
    )
    print(f"Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()
