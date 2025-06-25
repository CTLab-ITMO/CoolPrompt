from typing import Iterable
from langchain_core.language_models.base import BaseLanguageModel
from .logging_config import logger

METHODS = ["hype", "reflective"]
TASKS = ["classification", "generation"]


def validate_verbose(verbose):
    if verbose not in [0, 1, 2]:
        error_msg = f"Invalid verbose: {verbose}. Available values: 0, 1, 2."
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_model(model) -> None:
    logger.info(f"Validating model: {model}")
    if not isinstance(model, BaseLanguageModel):
        error_msg = "Model should be instance of LangChain BaseLanguageModel"
        logger.error(error_msg)
        raise TypeError(error_msg)


def validate_start_prompt(start_prompt):
    if not isinstance(start_prompt, str):
        if not start_prompt:
            error_msg = "Start prompt should be provided"
        else:
            error_msg = (
                "Start prompt should be a string. "
                f"Provided: {type(start_prompt).__name__}"
            )
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_task(task):
    if not isinstance(task, str):
        if not task:
            error_msg = "Task type should be provided"
        else:
            error_msg = (
                "Task type should be a string. "
                f"Provided: {type(task).__name__}"
            )
        logger.error(error_msg)
        raise ValueError(error_msg)
    if task not in TASKS:
        error_msg = (
            f"Invalid task type: {task}. "
            f"Available tasks: {', '.join(TASKS)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_dataset(dataset, target, method):
    print(f'Validating dataset {dataset} and target {target}')
    if dataset and not target:
        error_msg = "Dataset must be provided with the target"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if not isinstance(dataset, Iterable):
        error_msg = (
            "Start prompt should be an Iterable instance. "
            f"Provided: {type(dataset).__name__}"
        )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(dataset) <= 0:
            error_msg = (
                f"Dataset should be non-empty. Actual size: {len(dataset)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        if method == "reflective":
            error_msg = (
                "Train dataset is not defined "
                "for ReflectivePrompt optimization"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_target(target, dataset):
    if target:
        if not isinstance(target, Iterable):
            error_msg = (
                "Start prompt should be an Interable instance. "
                f"Provided: {type(target).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(target) != len(dataset):
            error_msg = (
                f"Dataset and target must have equal length. Actual "
                f"dataset size: {len(dataset)}, target size: {len(target)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_method(method):
    if not isinstance(method, str):
        error_msg = (
            "Method name should be a string. "
            f"Provided: {type(method).__name__}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    if method not in METHODS:
        error_msg = (
            f"Unsupported method: {method}. "
            f"Available methods: {', '.join(METHODS)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_problem_description(problem_description, method):
    if problem_description:
        if not isinstance(problem_description, str):
            error_msg = (
                "Start prompt should be a string. "
                f"Provided: {type(problem_description).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        if method == "reflective":
            error_msg = (
                "Problem description should be provided for "
                "ReflectivePrompt optimization"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_validation_size(validation_size):
    if not isinstance(validation_size, float) or not (
        0.0 <= validation_size <= 1.0
    ):
        error_msg = (
            "Validation size should be a float between 0.0 and 1.0. "
            f"Provided: {validation_size}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
