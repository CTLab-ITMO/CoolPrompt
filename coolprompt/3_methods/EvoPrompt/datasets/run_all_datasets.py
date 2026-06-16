"""Sequential launcher for EvoPrompt over all 8 datasets.

This script runs ``run.py`` one dataset after another with a fixed set of
hyperparameters (same for every dataset):

- ``popsize=10``           — population size
- ``budget=5``             — number of evolution iterations / epochs
- ``sample_num=50``        — train examples used to score candidate prompts
- ``test_sample_num=100``  — validation examples used in the final test step
- ``evo_mode=de``          — Differential Evolution (more stable than GA at small budget)
- ``initial=cot``          — initialise population from ``prompts.txt``
- ``initial_mode=topk``    — pick top-k by score on the dev sample
- ``template=v1``          — DE template from ``data.templates``
- ``model=gpt-5-nano``     — OpenAI chat model used through ``langchain_openai``
- ``temperature=1.0``      — sampling temperature for the chat model
- ``seed=5``               — reproducibility

Per-dataset optimisation metric (passed via ``--metric``):

- ``gsm8k``     → ``exact_match``
- ``tweeteval`` → ``f1_mera``
- everything else (``squad_v2``, ``common_gen``, ``xsum``, ``mediqa``,
  ``code_to_text``, ``concode``) → ``bert_score``

Each dataset gets its own output directory ``outputs/<dataset>_de_<metric>/``
and its own JSON optimisation log ``optimization_log.json`` inside that
directory. The JSON log stores the ``dataset`` and ``metric`` fields both at
the top level and inside the ``final`` block, next to the best prompt and
its dev / test scores.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from getpass import getpass


DATASETS = [
    "squad_v2",
    "gsm8k",
    "common_gen",
    "xsum",
    "tweeteval",
    "mediqa",
    "code_to_text",
    "concode",
]

# Per-dataset optimisation metric:
#   gsm8k     -> exact_match
#   tweeteval -> f1_mera
#   all other -> bert_score
DEFAULT_METRICS = {
    "squad_v2":     "bert_score",
    "gsm8k":        "exact_match",
    "common_gen":   "bert_score",
    "xsum":         "bert_score",
    "tweeteval":    "f1_mera",
    "mediqa":       "bert_score",
    "code_to_text": "bert_score",
    "concode":      "bert_score",
}


def build_command(dataset: str) -> list[str]:
    metric = DEFAULT_METRICS[dataset]
    output_dir = Path("outputs") / f"{dataset}_de_{metric}"
    results_json = output_dir / "optimization_log.json"
    return [
        sys.executable,
        "run.py",
        "--dataset", dataset,
        "--metric", metric,
        "--evo_mode", "de",
        "--popsize", "10",
        "--budget", "5",
        "--sample_num", "50",
        "--test_sample_num", "100",
        "--seed", "5",
        "--initial", "cot",
        "--initial_mode", "topk",
        "--template", "v1",
        "--model", "openai/gpt-5-nano",
        "--temperature", "1.0",
        "--output", str(output_dir),
        "--results_json", str(results_json),
    ]


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

    Path("outputs").mkdir(exist_ok=True)

    summary: list[tuple[str, int]] = []

    for dataset in DATASETS:
        cmd = build_command(dataset)
        print("\n" + "=" * 80)
        print(f"[run_all_datasets] Starting dataset: {dataset} "
              f"(metric={DEFAULT_METRICS[dataset]})")
        print("[run_all_datasets] Command:")
        print(" ".join(cmd))
        print("=" * 80)

        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        summary.append((dataset, completed.returncode))

        if completed.returncode != 0:
            print(
                f"[run_all_datasets] Dataset {dataset} failed with exit code "
                f"{completed.returncode}. Stopping sequence.",
                file=sys.stderr,
            )
            break

    print("\n" + "#" * 80)
    print("[run_all_datasets] Summary")
    print("#" * 80)
    for dataset, code in summary:
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"- {dataset}: {status}")

    return 0 if summary and all(code == 0 for _, code in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
