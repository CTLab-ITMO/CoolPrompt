"""Prompts scoring instrument. You can run it with
python prompts_scoring.py --input-file-path `input_file_path`
--output-file-path `output_file_path` --full --gen-only
Where:
    input_file_path: path to the input file. It must be a
        JSON with the structure {task_name: prompt}.
    output_file_path: path to the output file. Output will be
        structured as JSON: {task_name: {'score': score, 'prompt': prompt}}
        for each task and prompt.
    full: optional flag for using the full dataset.
    gen_only: optional flag for evaluating only generation tasks.

Check `prompts_scoring_example.ipynb` notebook for more examples.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

from src.utils.load_dataset_iterable import (
    GENERATION_TASKS,
    load_dataset_iterable,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run prompts scoring.")

parser.add_argument(
    "--gen-only", action="store_true", help="Skip classification tasks"
)
parser.add_argument("--input-file-path", required=True)
parser.add_argument("--output-file-path", required=True)
parser.add_argument(
    "--full", action="store_true", help="Enable evaluation on full dataset"
)
parser.add_argument(
    "--generation-metric",
    default="meteor",
    help="Generation metric. Must be one of bleu, rouge, meteor",
)
parser.add_argument(
    "--classification-metric",
    default="f1",
    help="Classification metric. Must be one of f1, accuracy",
)
args = parser.parse_args()
logger.info(f"Given args: {args}")

with open(args.input_file_path, "r") as f:
    prompts_json = json.load(f)

output_path = Path(args.output_file_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

from src.prompts_scoring.model_loader import ModelLoader

loader = ModelLoader(verbose=2)

result = {}

for task_name, prompt in prompts_json.items():

    logger.debug(f"Initializing task {task_name}")

    task_type, metric = (
        ("generation", args.generation_metric)
        if task_name in GENERATION_TASKS
        else ("classification", args.classification_metric)
    )

    if args.gen_only and task_type == "classification":
        continue

    loader.initialize(task_type, metric)

    logger.info(f"Loading test dataset {task_name}, full = {args.full}")
    dataset, target = load_dataset_iterable(
        dataset_name=task_name,
        split="test",
        sample_size=100 if not args.full else None,
    )

    logger.info(
        (
            f"Starting scoring {task_name}, "
            f"evaluator: {type(loader.evaluator).__name__}, "
            f"prompt: {prompt}"
        )
    )

    score = loader.get_metrics(
        candidate=prompt, dataset=dataset, target=target
    )

    result[task_name] = {
        "metric": {"name": metric, "score": score},
        "prompt": prompt,
    }

    logger.info(result)
    logger.info(f"Processed prompt for task: {task_name}")

with open(output_path, "w") as output_file:
    json.dump(result, output_file, indent=4)

logger.info(
    f"Evaluation completed. Results written to {args.output_file_path}"
)
