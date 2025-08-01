import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Iterable

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

from src.utils.load_dataset_iterable import (
    GENERATION_TASKS,
    load_dataset_iterable,
)
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.prompt_templates.basic_methods_templates import (
    ADAPT_TEMPLATE,
    IMPLEMENT_TEMPLATE,
    ROLE_EXTRACTING_TEMPLATE,
    SELECT_TEMPLATE,
)


BASIC_PROMPTING_METHODS = [
    "zero-shot",
    "few-shot",
    "role-based",
    "few-shot-chain-of-thoughts",
    "zero-shot-chain-of-thoughts",
    "self-discover",
]

FEW_SHOT_COT_MAX_TOKENS = 100
ROLE_BASED_MAX_TOKENS = 50
SELF_DISCOVER_MAX_TOKENS = 125


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


def run_zero_shot(prompts: dict[str, str]) -> None:
    """Gets basic prompts from provided `prompts` and returns a
    dict with {task: prompt} for each task and prompt.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
    """

    return prompts


def run_few_shot(
    prompts: dict[str, str],
    num_shots: int = 3,
):
    """Converts basic prompts from `prompts` to few-shot form based on
    `num_shot` samples from datasets, then returns a dict.
    Taken from DistillPrompt realization.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
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
        few_shot_prompt = prompt + "\n\nExamples:\n" + generate_samples(task)
        result[task] = few_shot_prompt

    return result


def run_role_based(
    prompts: dict[str, str],
    model: BaseLanguageModel,
):
    """Gets basic prompts from `prompts` and rewrites them using the role,
    which is extracted by the query to `model` LLM with special template.
    Then returns a dict.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        model (BaseLanguageModel): LangChain LLM.
    """

    result = {}

    def extract_role(prompt):
        response = model.invoke(
            ROLE_EXTRACTING_TEMPLATE.format(instruction=prompt)
        ).strip()
        return response[
            (response.rfind("<ROLE>") + len("<ROLE>")) : response.rfind(
                "</ROLE>"
            )
        ]

    for task, prompt in prompts.items():
        role = extract_role(prompt)
        role_based_prompt = "Your role is " + role + ". " + prompt
        result[task] = role_based_prompt

    return result


def run_few_shot_chain_of_thoughts(
    prompts: dict[str, str],
    labels: Iterable,
    model: BaseLanguageModel,
    num_shots: int = 3,
):
    """Converts basic prompts from `prompts` to few-shot form with shown
    chain-of-thought based on `num_shot` samples from datasets,
    then return a dict.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
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
                template = (
                    CLASSIFICATION_TASK_TEMPLATE.format(
                        PROMPT=prompt, LABELS=labels_list, INPUT=input
                    )
                    + " Let's think step by step"
                )
            else:
                template = (
                    GENERATION_TASK_TEMPLATE.format(PROMPT=prompt, INPUT=input)
                    + " Let's think step by step"
                )
            output = model.invoke(template)

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

    return result


def run_zero_shot_chain_of_thoughts(prompts: dict[str, str]):
    """Gets basic prompts from `prompts` and appends
    '\nLet's think step by step.' to each one. Then returns a dict.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
    """

    result = {}

    for task, prompt in prompts.items():
        zero_shot_cot_prompt = prompt + "\nLet's think step by step."
        result[task] = zero_shot_cot_prompt

    return result


def run_self_discover(prompts: dict[str, str], model: BaseLanguageModel):
    """Gets basic prompts from `prompts` and performs a
    self-discover algorithm using provided `model` LLM.
    Then returns a dict.

    Args:
        prompts (dict[str, str]): dict with task name as a key and
            corresponding basic prompt as a value.
        model (BaseLanguageModel): LangChain LLM.
    """

    result = {}

    for task, prompt in prompts.items():
        selected_modules = model.invoke(SELECT_TEMPLATE.format(Task=prompt))
        adapted_modules = model.invoke(
            ADAPT_TEMPLATE.format(
                Task=prompt, selected_modules=selected_modules
            )
        )
        implement_prompt = model.invoke(
            IMPLEMENT_TEMPLATE.format(
                Task=prompt, adapted_modules=adapted_modules
            )
        )

        result[task] = implement_prompt

    return result


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
        help=(
            "Method of prompting. "
            f"Must be one of {','.join(BASIC_PROMPTING_METHODS)}"
        ),
    )
    parser.add_argument(
        "--output-file-path", required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--input-file-path",
        default="basic_prompts.json",
        help="Path to the input JSON file (default: basic_prompts.json)",
    )
    parser.add_argument(
        "--num-shots",
        default=3,
        help="Number of examples for few-shot methods",
    )
    parser.add_argument(
        "--labels-file-path",
        default="labels.json",
        help="Path to the labels JSON file (default: labels.json)",
    )

    args = parser.parse_args()

    prompts = load_prompts(args.input_file_path)

    match args.method:
        case "zero-shot":
            result = run_zero_shot(prompts)
        case "few-shot":
            result = run_few_shot(prompts, args.num_shots)
        case "few-shot-chain-of-thoughts":
            labels = load_labels(args.labels_file_path)
            model = DefaultLLM.init(
                langchain_config={"max_new_tokens": FEW_SHOT_COT_MAX_TOKENS}
            )
            result = run_few_shot_chain_of_thoughts(
                prompts, labels, model, args.num_shots
            )
        case "role-based":
            model = DefaultLLM.init(
                langchain_config={"max_new_tokens": ROLE_BASED_MAX_TOKENS}
            )
            result = run_role_based(prompts, model)
        case "zero-shot-chain-of-thoughts":
            result = run_zero_shot_chain_of_thoughts(prompts)
        case "self-discover":
            model = DefaultLLM.init(
                langchain_config={"max_new_tokens": SELF_DISCOVER_MAX_TOKENS}
            )
            result = run_self_discover(prompts, model)

    with open(args.output_file_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
