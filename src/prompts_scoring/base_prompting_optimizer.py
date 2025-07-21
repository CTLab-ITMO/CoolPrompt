"""Basic prompting methods optimizer.
Available methods: `basic` (returns basic prompts)

You can run it as python base_prompting_optimizer.py --method `method`
--input-file-path `input_file_path` --output-file-path `output_file_path`
Where:
    `method`: method name, must be one of 'basic.'
    `input_file_path`: path to the input json file with basic prompts.
    Expected structure {task_name: prompt, ...}.
    Defaults to 'basic_prompts.json'.
    `output_file_path`: path to the output file. Output will be structured
    as json lines for each task and prompt: {'task': task, 'prompt': prompt}.
"""

import argparse
import json
from pathlib import Path


def load_prompts(input_file="basic_prompts.json"):
    """Loads prompts from json file

    Args:
        input_file (str): path to input file with basic prompts.
        Expected structure: {task_name: prompt, ...}.
        Defaults to 'basic_prompts.json'
    """
    with open(input_file) as f:
        return json.load(f)


def run_base_prompts(prompts: dict[str, str], output_file_path: str | Path):
    """Gets basic prompts from provided `prompts` and for each task and prompt
    writes to the `output_file_path` by lines with a
    structure {'task': task, 'prompt': prompt}

    Args:
        prompts: dict with task name as a key and corresponding
        basic prompt as a value
        output_file_path: path to the output file"""
    with open(output_file_path, "a") as f:
        for task, prompt in prompts.items():
            f.write(
                "".join([json.dumps({"task": task, "prompt": prompt}), "\n"])
            )
            f.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process prompts and save to "
            "file as {'task': task, 'prompt': prompt} lines."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="Method of prompting. Must be one of 'basic'",
    )
    parser.add_argument(
        "--output-file-path", required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--input-file-path",
        default="basic_prompts.json",
        help="Path to the input JSON file (default: basic_prompts.json)",
    )
    args = parser.parse_args()
    prompts = load_prompts(args.input_file_path)
    match args.method:
        case "basic":
            run_base_prompts(prompts, args.output_file_path)


if __name__ == "__main__":
    main()
