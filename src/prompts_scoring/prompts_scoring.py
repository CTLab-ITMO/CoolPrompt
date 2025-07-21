"""Prompts scoring instrument. You can run it with
python prompts_scoring.py --input-file-path `input_file_path`
--output-file-path `output_file_path` --full --gen-only
Where:
    input_file_path: path to the input file. It must be structured
    as json lines {'task': task, 'prompt': prompt}.
    output_file_path: path to the output file. Output will be
    structured as json lines {'task': task, 'score': score, 'prompt': prompt}.
    Writes via appending line by line.
    full: optional flag for using the full dataset.
    gen_only: optional flag for evaluating only generation tasks.
"""

import argparse
import json
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run prompts scoring.")

parser.add_argument(
    "--gen_only", action="store_true", help="Skip classification tasks"
)
parser.add_argument("--input-file-path", required=True)
parser.add_argument("--output-file-path", required=True)
parser.add_argument(
    "--full", action="store_true", help="Enable evaluation on full dataset"
)
args = parser.parse_args()
logger.info(f"Given args: {args}")

from src.evaluation.evaluator import GenerationEvaluator
from src.prompts_scoring.model_loader import ModelLoader

loader = ModelLoader("T-tech/T-lite-it-1.0", verbose=2)

with open(args.output_file_path, "a") as output_file:
    with open(args.input_file_path, "r") as input_file:
        for line in input_file:
            try:
                try:
                    data = json.loads(line.strip(), strict=False)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse line as JSON: {line}")
                    continue

                task = data.get("task")
                prompt = data.get("prompt")

                logger.debug(f"Initializing task {task}")
                loader.initialize(task)
                if args.gen_only and not isinstance(
                    loader.evaluator, GenerationEvaluator
                ):
                    continue
                if prompt == "":
                    prompt = loader.base_prompt
                logger.info(
                    (
                        f"Starting scoring {task}, "
                        f"evaluator: {type(loader.evaluator).__name__}, "
                        f"prompt: {prompt}"
                    )
                )

                score = loader.get_metrics(
                    candidate=prompt, split="test", full=args.full
                )

                result = {
                    "task": task,
                    "score": score,
                    "prompt": prompt,
                }
                logger.info(result)
                output_file.write(json.dumps(result) + "\n")
                output_file.flush()
                logger.info(f"Processed prompt for task: {task}")
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")

    logger.info(
        f"Evaluation completed. Results written to {args.output_file_path}"
    )
    loader.destroy()
