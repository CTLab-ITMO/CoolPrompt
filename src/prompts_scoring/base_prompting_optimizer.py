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
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.prompt_templates.basic_prompting_methods_templates import (
    ROLE_EXTRACTING_TEMPLATE,
)
from src.utils.load_dataset_iterable import (
    GENERATION_TASKS,
    load_dataset_iterable,
)

BASIC_PROMPTING_METHODS = [
    "zero-shot",
    "few-shot",
    "role-based",
    "few-shot-chain-of-thoughts",
    "zero-shot-chain-of-thoughts",
]


def load_prompts(input_file: str | Path = "basic_prompts.json"):
    """Loads prompts from JSON file

    Args:
        input_file (str | Path): path to input file with basic prompts.
            Expected structure: JSON with {task_name: prompt} objects
            for each task and prompt. Defaults to 'basic_prompts.json'
    """
    with open(input_file) as f:
        return json.load(f)


def load_labels(input_file: str | Path = "labels.json"):
    """Loads labels from JSON file

    Args:
        input_file (str | Path): path to input file with task labels.
            Expected structure: JSON with {task_name: labels_list} objects
            for each task and labels list. Defaults to 'labels.json'
    """
    with open(input_file) as f:
        return json.load(f)


def get_labels(labels: list[str], task: str):
    """Gets labels for the given task

    Args:
        labels (list[str]): list of labels for the given task
        task (str): task name (ex. mnli or bbh/word_sorting).
    Raises:
        ValueError: if provided task is unknown.
    """

    labels_list = labels.get(task)
    if labels_list is None:
        raise ValueError(
            f"Task {task} is not known or not a classification task"
        )
    else:
        return labels_list


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
    num_shots: int = 3,
):
    """Converts basic prompts from `prompts` to few-shot form based on
    `num_shot` samples from datasets, then writes it to
    `output_file_path` as JSON. Taken from DistillPrompt realization.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        output_file_path (str | Path): path to the output file.
        num_shot (int): number of samples will be taken from dataset.
            Defaults to 3.
    """

    result = {}

    def generate_samples(task):

        dataset, target = load_dataset_iterable(
            dataset_name=task,
            split="train",
            sample_size=num_shots,
        )

        samples = list(zip(dataset, target))

        random.shuffle(samples)

        formatted_string = ""
        for i, (input, output) in enumerate(samples):
            formatted_string += f"Example {i + 1}:\n"
            formatted_string += (
                f'Text: "{input.strip()}"\nAnswer: {output}\n\n'
            )

        return formatted_string

    for task, prompt in prompts.items():
        few_shot_prompt = prompt + "\n\nExamples:\n" + generate_samples()
        result[task] = few_shot_prompt

    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


def run_role_based(
    prompts: dict[str, str],
    output_file_path: str | Path,
    model: BaseLanguageModel,
):
    """Gets basic prompts from `prompts` and rewrites them using the role,
    which is extracted by the query to `model` LLM with special template.
    Then writes to `output_file_path` as JSON.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        output_file_path (str | Path): path to the output file.
        model (BaseLanguageModel): LangChain LLM.
    """

    result = {}

    def extract_role(prompt):
        response = model(ROLE_EXTRACTING_TEMPLATE.format(instruction=prompt))
        return response.strip()

    for task, prompt in prompts.items():
        role = extract_role(prompt)
        role_based_prompt = "Your role is " + role + ". " + prompt
        result[task] = role_based_prompt

    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


def run_few_shot_chain_of_thoughts(
    prompts: dict[str, str],
    output_file_path: str | Path,
    labels: Iterable,
    model: BaseLanguageModel,
    num_shots: int = 3,
):
    """Converts basic prompts from `prompts` to few-shot form with shown
    chain-of-thought based on `num_shot` samples from datasets,
    then writes it to `output_file_path` as JSON.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        output_file_path (str | Path): path to the output file.
        labels (Iterable): list of labels for classification tasks
        num_shot (int): number of samples will be taken from dataset.
            Defaults to 3.
        model (BaseLanguageModel): LangChain LLM.
    """

    result = {}

    def generate_samples(task, prompt):

        samples, _ = load_dataset_iterable(
            dataset_name=task,
            split="train",
            sample_size=num_shots,
        )

        random.shuffle(samples)

        formatted_string = ""
        for i, input in enumerate(samples):

            if task not in GENERATION_TASKS:
                labels_list = get_labels(labels, task)
                template = CLASSIFICATION_TASK_TEMPLATE.format(
                    PROMPT=prompt, LABELS=labels_list, INPUT=input
                )
            else:
                template = GENERATION_TASK_TEMPLATE.format(
                    PROMPT=prompt, INPUT=input
                )
            output = model(template)

            formatted_string += f"Example {i + 1}:\n"
            formatted_string += (
                f'Text: "{input.strip()}"\nAnswer: {output}\n\n'
            )

        return formatted_string

    for task, prompt in prompts.items():
        few_shot_prompt = (
            prompt + "\n\nExamples:\n" + generate_samples(task, prompt)
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
        help=f"Method of prompting. Must be one of {','.join(BASIC_PROMPTING_METHODS)}",
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
