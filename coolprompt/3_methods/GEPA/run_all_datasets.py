"""Sequential launcher for GEPA optimisation over all 8 supported datasets.

Runs ``run.py`` once per dataset with a fixed set of hyperparameters:

- ``--task_lm openai/gpt-4o-mini``       — model used for the task
- ``--reflection_lm openai/gpt-4o-mini`` — model used for reflection / mutation
- ``--max_metric_calls 150``             — optimisation budget per dataset
- ``--train_size 50``                    — examples in GEPA trainset
- ``--val_size 100``                     — examples in GEPA valset
- ``--test_size 100``                    — examples for final test evaluation
- ``--seed 5``                           — reproducibility seed

Per-dataset optimisation metric:
- ``gsm8k``     → ``exact_match``
- ``tweeteval`` → ``f1_mera``
- all others    → ``bert_score``

Each dataset gets its own output directory ``outputs/<dataset>/`` and a
JSON results file ``outputs/<dataset>/results.json``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run import DEFAULT_METRICS, DATASETS


ROOT = Path(__file__).resolve().parent
PYTHON_BIN = Path(sys.executable)
RUN_SCRIPT = ROOT / "run.py"

# ── Model configuration ────────────────────────────────────────────────────
TASK_LM = "openai/gpt-4o-mini"
REFLECTION_LM = "openai/gpt-4o-mini"
# API_KEY = ""
API_KEY = ""
# BASE_URL = "https://foundation-models.api.cloud.ru/v1"
BASE_URL = "https://openrouter.ai/api/v1"

# ── Model call settings ───────────────────────────────────────────────────
MAX_TOKENS = 4000
TIMEOUT = 60
MAX_RETRIES = 2
REQUESTS_PER_SECOND = 3.0

# ── Optimisation hyperparameters ───────────────────────────────────────────
TOKEN_BUDGET = 80000   # set to e.g. 200_000 to use token budget instead of metric calls
POPULATION_SIZE = 5
NUM_EPOCHS = None      # ignored when TOKEN_BUDGET is set; set to e.g. 5 for metric-call mode
TRAIN_SIZE = 50
VAL_SIZE = 100
TEST_SIZE = 10
SEED = 158


def build_command(dataset: str) -> list[str]:
    metric = DEFAULT_METRICS[dataset]
    output_dir = ROOT / "outputs" / dataset
    results_json = output_dir / "results.json"
    cmd = [
        str(PYTHON_BIN),
        str(RUN_SCRIPT),
        "--dataset", dataset,
        "--metric", metric,
        "--task_lm", TASK_LM,
        "--reflection_lm", REFLECTION_LM,
        "--train_size", str(TRAIN_SIZE),
        "--val_size", str(VAL_SIZE),
        "--test_size", str(TEST_SIZE),
        "--seed", str(SEED),
        "--max_tokens", str(MAX_TOKENS),
        "--timeout", str(TIMEOUT),
        "--max_retries", str(MAX_RETRIES),
        "--requests_per_second", str(REQUESTS_PER_SECOND),
        "--output", str(output_dir),
        "--results_json", str(results_json),
    ]
    if POPULATION_SIZE is not None:
        cmd += ["--population_size", str(POPULATION_SIZE)]
    # num_epochs has no effect when token_budget is active
    if TOKEN_BUDGET is None and NUM_EPOCHS is not None:
        cmd += ["--num_epochs", str(NUM_EPOCHS)]
    if API_KEY:
        cmd += ["--api_key", API_KEY]
    if BASE_URL:
        cmd += ["--base_url", BASE_URL]
    if TOKEN_BUDGET is not None:
        cmd += ["--token_budget", str(TOKEN_BUDGET)]
    return cmd


def main() -> int:
    if not API_KEY and not os.getenv("OPENAI_API_KEY"):
        print(
            "[run_all_datasets] ERROR: set API_KEY in this file "
            "or export OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    (ROOT / "outputs").mkdir(exist_ok=True)

    summary: list[tuple[str, int]] = []

    for dataset in ["squad_v2", "gsm8k", "common_gen", "xsum", "tweeteval"]:
        cmd = build_command(dataset)
        metric = DEFAULT_METRICS[dataset]

        print("\n" + "=" * 80)
        print(
            f"[run_all_datasets] Starting dataset: {dataset} "
            f"(metric={metric})"
        )
        print("[run_all_datasets] Command:")
        print(" ".join(cmd))
        print("=" * 80)

        completed = subprocess.run(cmd, cwd=ROOT)
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
        print(f"  - {dataset}: {status}")

    return 0 if summary and all(code == 0 for _, code in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
