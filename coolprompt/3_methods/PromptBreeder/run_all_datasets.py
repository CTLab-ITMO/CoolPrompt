"""Sequential launcher for PromptBreeder optimization on all supported datasets.

This script runs [`main.py`](main.py) once per dataset using the requested
hyperparameters:

- population size = 10 via `-mp 2 -ts 5`
- optimization iterations = 5 via `-n 5`
- train examples = 50 via `-e 50`
- model = `gpt-5-nano`
- temperature = `1.0`
- metric: per-dataset (gsm8k -> exact_match, tweeteval -> f1_mera,
  all others -> bert_score), resolved via
  [`default_metric_for`](pb/metrics.py:194)

Each run writes its own JSON log into [`runs/`](runs).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from getpass import getpass

from dotenv import load_dotenv

from pb.datasets import SUPPORTED_DATASETS
from pb.metrics import default_metric_for


ROOT = Path(__file__).resolve().parent
MAIN_FILE = ROOT / "main.py"
PYTHON_BIN = Path(sys.executable)

NUM_MUTATION_PROMPTS = 2
NUM_THINKING_STYLES = 5
NUM_EVALS = 50
NUM_SIMULATIONS = 5
MODEL_NAME = "openai/gpt-5-nano"
TEMPERATURE = "1.0"


def build_command(dataset: str) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(MAIN_FILE),
        "-d",
        dataset,
        "-mp",
        str(NUM_MUTATION_PROMPTS),
        "-ts",
        str(NUM_THINKING_STYLES),
        "-e",
        str(NUM_EVALS),
        "-n",
        str(NUM_SIMULATIONS),
        "-m",
        MODEL_NAME,
        "-t",
        TEMPERATURE,
        "-M",
        default_metric_for(dataset),
    ]


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

    print("=" * 80)
    print("Starting sequential PromptBreeder runs for all datasets")
    print(f"Python: {PYTHON_BIN}")
    print(f"Main:   {MAIN_FILE}")
    print(f"Datasets: {', '.join(SUPPORTED_DATASETS)}")
    print(
        "Hyperparameters: "
        f"population={NUM_MUTATION_PROMPTS * NUM_THINKING_STYLES} "
        f"(-mp {NUM_MUTATION_PROMPTS} * -ts {NUM_THINKING_STYLES}), "
        f"num_evals={NUM_EVALS}, simulations={NUM_SIMULATIONS}, "
        f"model={MODEL_NAME}, temperature={TEMPERATURE}"
    )
    print("Per-dataset metrics:")
    for dataset in SUPPORTED_DATASETS:
        print(f"  - {dataset}: {default_metric_for(dataset)}")
    print("=" * 80)

    failed_datasets: list[str] = []

    for index, dataset in enumerate(SUPPORTED_DATASETS, start=1):
        command = build_command(dataset)
        print()
        print("#" * 80)
        print(f"[{index}/{len(SUPPORTED_DATASETS)}] Running dataset: {dataset}")
        print("Command:")
        print(" ".join(command))
        print("#" * 80)

        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )

        if result.returncode != 0:
            failed_datasets.append(dataset)
            print(f"Dataset {dataset} failed with exit code {result.returncode}")
        else:
            print(f"Dataset {dataset} finished successfully")

    print()
    print("=" * 80)
    if failed_datasets:
        print("Completed with failures.")
        print("Failed datasets:")
        for dataset in failed_datasets:
            print(f"- {dataset}")
        return 1

    print("All dataset runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
