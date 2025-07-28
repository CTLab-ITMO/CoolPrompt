"""Basic prompting methods optimizer.
Available methods:
    `basic` (returns basic prompts)

You can run it as python base_prompting_optimizer.py --method `method`
--input-file-path `input_file_path` --output-file-path `output_file_path`
Where:
    method: method name, must be one of 'basic.'
    input_file_path: path to the input json file with basic prompts.
        Expected structure {task_name: prompt, ...}.
        Defaults to 'basic_prompts.json'.
    output_file_path: path to the output file. Output will be structured
        as JSON: {task_name: prompt} for each task and prompt.

Check `prompts_scoring_example.ipynb` notebook for more examples.
"""

import argparse
import json
from pathlib import Path
import random
from typing import Iterable


def load_prompts(input_file: str | Path = "basic_prompts.json"):
    """Loads prompts from json file

    Args:
        input_file (str | Path): path to input file with basic prompts.
            Expected structure: JSON with {task_name: prompt} objects
            for each task and prompt. Defaults to 'basic_prompts.json'
    """
    with open(input_file) as f:
        return json.load(f)


def run_zero_shot(
    prompts: dict[str, str], output_file_path: str | Path
) -> None:
    """Gets basic prompts from provided `prompts` and for each task and prompt
    writes JSON to the `output_file_path` with a structure {task_name: prompt}
    for each task and prompt.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        output_file_path (str | Path): path to the output file.
    """

    result = {}

    for task, prompt in prompts.items():
        result[task] = prompt

    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


def run_few_shot(
    prompts: dict[str, str],
    output_file_path: str | Path,
    dataset: Iterable,
    target: Iterable,
    num_shots: int = 3,
):
    result = {}

    def generate_samples():

        samples = random.sample(list(zip(dataset, target)), num_shots)

        formatted_string = ""
        for i, (input, output) in enumerate(samples):
            formatted_string += f"Example {i + 1}:\n"
            formatted_string += f'Text: "{input.strip()}"\nLabel: {output}\n\n'

        return formatted_string

    for task, prompt in prompts.items():
        few_shot_prompt = "".join(
            [prompt, "\n\nExamples:\n", generate_samples()]
        )
        result[task] = few_shot_prompt

    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


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
