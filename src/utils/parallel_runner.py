"""Reusable parallel benchmark runner.

Runs a list of benchmark tasks either sequentially or in
parallel using threads.  Each task gets a fresh PromptTuner
to avoid mutable-state conflicts, while sharing a single
(thread-safe) LLM client.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from coolprompt.assistant import PromptTuner


@dataclass
class BenchmarkTask:
    """Method-agnostic descriptor for one benchmark task."""

    key: str
    start_prompt: str
    task: str
    metric: str
    problem_description: str
    loader: Callable[[], pd.DataFrame]
    train_loader: Callable[[], pd.DataFrame] | None = None
    run_kwargs: dict[str, Any] = field(
        default_factory=dict,
    )


class ParallelBenchmarkRunner:
    """Run benchmark tasks with optional parallelism.

    Args:
        llm: Shared LangChain LLM (thread-safe).
        output_path: Path for the results JSON file.
        max_workers: Thread count (1 = sequential).
        sample_fn: Optional transform applied to each
            task's DataFrame before running.
    """

    def __init__(
        self,
        llm,
        output_path: Path,
        max_workers: int = 1,
        sample_fn: Callable[
            [pd.DataFrame], pd.DataFrame
        ] | None = None,
    ):
        self._llm = llm
        self._output_path = output_path
        self._max_workers = max_workers
        self._sample_fn = sample_fn

        self._lock = threading.Lock()
        self._results: dict = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run_all(
        self,
        tasks: list[BenchmarkTask],
    ) -> dict:
        """Execute *tasks*, skipping already-completed ones.

        Returns:
            The full results dict (including resumed ones).
        """
        self._output_path.parent.mkdir(
            parents=True, exist_ok=True,
        )
        if self._output_path.exists():
            with open(self._output_path) as f:
                self._results = json.load(f)

        pending = [
            t for t in tasks
            if t.key not in self._results
        ]
        skipped = len(tasks) - len(pending)
        if skipped:
            print(
                f"Resuming: {skipped} tasks already done, "
                f"{len(pending)} remaining"
            )

        print(
            f"Running {len(pending)} tasks "
            f"(workers: {self._max_workers})"
        )

        if self._max_workers <= 1:
            self._run_sequential(pending)
        else:
            self._run_parallel(pending)

        print(
            f"\nAll done. Results saved to "
            f"{self._output_path}"
        )
        return self._results

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _run_sequential(
        self,
        tasks: list[BenchmarkTask],
    ):
        for task in tasks:
            self._run_one(task)

    def _run_parallel(
        self,
        tasks: list[BenchmarkTask],
    ):
        with ThreadPoolExecutor(
            max_workers=self._max_workers,
        ) as pool:
            futures = {
                pool.submit(self._run_one, t): t
                for t in tasks
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(
                        f"UNHANDLED error on "
                        f"{task.key}: {e}"
                    )

    def _run_one(self, task: BenchmarkTask):
        """Run a single benchmark task end-to-end."""
        print(f"\n{'='*50}")
        print(f"Running: {task.key}")
        print(f"{'='*50}")

        # Load data
        try:
            df = task.loader()
            if self._sample_fn is not None:
                df = self._sample_fn(df)
            dataset = list(df["input_data"])
            target = list(df["target"])

            # Load separate training split if available
            train_dataset = None
            train_targets = None
            if task.train_loader is not None:
                train_df = task.train_loader()
                if self._sample_fn is not None:
                    train_df = self._sample_fn(train_df)
                train_dataset = list(
                    train_df["input_data"]
                )
                train_targets = list(train_df["target"])
        except Exception as e:
            print(f"ERROR loading {task.key}: {e}")
            self._record(task.key, {"error": str(e)})
            return

        # Fresh PromptTuner per task (mutable state)
        pt = PromptTuner(self._llm)

        # Build extra kwargs, removing 'method' from
        # run_kwargs and adding train data if available
        extra_kwargs = {
            k: v
            for k, v in task.run_kwargs.items()
            if k != "method"
        }
        if train_dataset is not None:
            extra_kwargs["train_dataset"] = train_dataset
            extra_kwargs["train_targets"] = train_targets

        try:
            final_prompt = pt.run(
                task.start_prompt,
                task.task,
                dataset,
                target,
                task.run_kwargs.get("method", "pe2"),
                task.metric,
                task.problem_description,
                **extra_kwargs,
            )
        except Exception as e:
            print(f"ERROR on {task.key}: {e}")
            self._record(task.key, {"error": str(e)})
            return

        entry = {
            "metric": {
                "name": task.metric,
                "start_score": pt.init_metric,
                "final_metric": pt.final_metric,
            },
            "prompt": final_prompt,
        }
        self._record(task.key, entry)
        print(
            f"Done {task.key}: "
            f"{pt.init_metric:.4f} -> {pt.final_metric:.4f}"
        )

    def _record(self, key: str, value: dict):
        """Thread-safe: store result and persist to disk."""
        with self._lock:
            self._results[key] = value
            self._save()

    def _save(self):
        """Atomically write results JSON."""
        tmp = self._output_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._results, f, indent=2)
        tmp.rename(self._output_path)
