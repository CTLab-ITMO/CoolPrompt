from typing import Any, Iterable, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.logging_config import logger
from coolprompt.utils.enums import Method, Task


def validate_verbose(verbose: int) -> None:
    """Checks that the provided verbose parameter is either 0, 1 or 2.

    Args:
        verbose (int): Provided verbose parameter.
    Raises:
        ValueError: If `verbose` is neither 0, 1 or 2."""

    if verbose not in [0, 1, 2]:
        error_msg = f"Invalid verbose: {verbose}. Available values: 0, 1, 2."
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_model(model: BaseLanguageModel) -> None:
    """Checks that the provided model is a
    LangChain BaseLanguageModel instance.

    Args:
        model (BaseLanguageModel): Provided model.
    Raises:
        TypeError: If `model` is not an instance of
        LangChain BaseLanguageModel.
    """

    if not isinstance(model, BaseLanguageModel):
        error_msg = (
            "Provided model must be an "
            "instance of LangChain BaseLanguageModel"
        )
        logger.error(error_msg)
        raise TypeError(error_msg)


def validate_start_prompt(start_prompt: str) -> None:
    """Checks that the start prompt is provided as a string.

    Args:
        start_prompt (str): Provided start prompt.
    Raises:
        TypeError: If `start_prompt` is not a string."""

    if not isinstance(start_prompt, str):
        if not start_prompt:
            error_msg = "Start prompt must be provided."
        else:
            error_msg = (
                "Start prompt must be a string. "
                f"Provided: {type(start_prompt).__name__}."
            )
        logger.error(error_msg)
        raise TypeError(error_msg)


def validate_task(task: str) -> Task:
    """Checks that a valid task type is provided.

    Args:
        task (str): Provided task type. Must be one of:
            ["classification", "generation"].
    Returns:
        Task: The validated task type.
    Raises:
        TypeError: If `task` is not a string.
        ValueError: If `task` is not one of
            ["classification", "generation"]."""

    if not isinstance(task, str):
        if not task:
            error_msg = "Task type must be provided."
        else:
            error_msg = (
                "Task type must be a string. "
                f"Provided: {type(task).__name__}."
            )
        logger.error(error_msg)
        raise TypeError(error_msg)
    if task not in Task._value2member_map_:
        error_msg = (
            f"Invalid task type: {task}. "
            f"Available tasks: {', '.join(list(
                Task._value2member_map_.keys()))}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return Task(task)


def validate_dataset(
    dataset: Iterable | None, target: Iterable | None, method: Method
) -> None:
    """Checks that the provided dataset is an Iterable instance
    and the target is also provided. Also checks that the dataset is
    provided if the method is data-driven.

    Args:
        dataset (Iterable | None): Provided dataset.
        target (Iterable | None): Provided target.
        method (Method): Provided method.
    Raises:
        TypeError: If `dataset` is not None but is not Iterable.
        ValueError: If `dataset` is None but `method` requiers a dataset,
            or if `dataset` is provided but `target` is None.
    """

    if dataset is not None:
        if target is None:
            error_msg = "Dataset must be provided with the target."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(dataset, Iterable):
            error_msg = (
                "Dataset must be an Iterable instance. "
                f"Provided: {type(dataset).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        if len(dataset) == 0:
            if method.is_data_driven():
                error_msg = (
                    "Dataset must be non-empty when using data-driven "
                    f"optimization method '{method}'. You can try using HyPE "
                    "optimization ('hype' as method parameter) which "
                    "does not require any train dataset."
                )
            else:
                error_msg = (
                    "Dataset must be non-empty for evaluation when using "
                    f"'{method}' optimization method. If you do not want to "
                    "evaluate your prompts, please do not provide any dataset."
                )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        if method.is_data_driven():
            error_msg = (
                "Train dataset must be provided for data-driven "
                f"optimization method '{method}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_target(target: Iterable | None, dataset: Iterable | None) -> None:
    """Checks that the provided target is an Iterable instance
    with the same length as the provided dataset.

    Args:
        target (Iterable | None): Provided target.
        dataset (Iterable | None): Provided dataset. Can not be None if
            `target` is not None.
    Raises:
        TypeError: If `target` is not Iterable.
        ValueError: If `target` length does not equal the `dataset` length,
            or if `dataset` is None while `target` is not."""

    if target is not None:
        if dataset is None:
            error_msg = "Dataset cannot be None if target is provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(target, Iterable):
            error_msg = (
                "Target must be an Interable instance. "
                f"Provided: {type(target).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        if len(target) != len(dataset):
            error_msg = (
                f"Dataset and target must have equal length. Actual "
                f"dataset size: {len(dataset)}, target size: {len(target)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_method(method: str) -> Method:
    """Checks that a valid method name is provided.

    Args:
        method (str): Provided method. Must be one of:
            ["hype", "reflective", "distill"].
    Returns:
        Method: The validated method.
    Raises:
        TypeError: If `method` is not a string.
        ValueError: If `method` is not one of
            ["hype", "reflective", "distill"].
    """

    if not isinstance(method, str):
        error_msg = (
            "Method name must be a string. "
            f"Provided: {type(method).__name__}."
        )
        logger.error(error_msg)
        raise TypeError(error_msg)
    if method not in Method._value2member_map_:
        error_msg = (
            f"Unsupported method: {method}. "
            f"Available methods: {', '.join(list(
                Method._value2member_map_.keys()))}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return Method(method)


def validate_problem_description(
    problem_description: str | None, method: Method
) -> None:
    """Checks that the problem description is provided as a string
    when using the ReflectivePrompt optimization.

    Args:
        problem_description (str | None): Provided problem description.
        method (Method): Provided method.
    Raises:
        TypeError: If `problem_description` is not a string.
        ValueError: If `problem_description` is not provided when
            using the ReflectivePrompt method.
    """

    if problem_description is not None:
        if not isinstance(problem_description, str):
            error_msg = (
                "Problem description must be a string. "
                f"Provided: {type(problem_description).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
    else:
        if method is Method.REFLECTIVE:
            error_msg = (
                "Problem description must be provided for "
                "ReflectivePrompt optimization."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_validation_size(validation_size: float | Any) -> None:
    """Checks that the provided validation_size is a float from 0.0 to 1.0.

    Args:
        validation_size (float): Provided validation size.
    Raises:
        ValueError: If `validation_size` is not a float in [0.0, 1.0]."""

    if not isinstance(validation_size, float) or not (
        0.0 <= validation_size <= 1.0
    ):
        error_msg = (
            "Validation size must be a float between 0.0 and 1.0. "
            f"Provided: {validation_size}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_run(
    start_prompt: str,
    task: str,
    dataset: Iterable | None,
    target: Iterable | None,
    method: str,
    problem_description: str | None,
    validation_size: float,
) -> Tuple[Task, Method]:
    """Checks if args for PromptTuner.run() are valid.

    Args:
        start_prompt (str): Provided start prompt. Must be a string.
        task (str): Provided task type. Must be one of:
            ["classification", "generation"].
        dataset (Iterable | None): Provided dataset.
            Required for data-driven methods.
        target (Iterable | None): Provided target labels for dataset.
            Required if dataset provided.
        method (str): Provided method. Must be one of:
            ["hype", "reflective", "distill"].
        problem_description (str | None): Provided problem description.
            Must be a string, required when using the ReflectivePrompt method.
        validation_size (float): Provided validation size.
            Must be a float in [0.0, 1.0].
    Returns:
        Tuple[Task, Method]: The validated task and method.
    Raises:
        TypeError: If any argument has incorrect type:
            -`start_prompt` is not a string
            -`task` is not a string.
            -`dataset` is not None but is not Iterable
            -`dataset` is provided but target is None
            -`target` is not Iterable
            -`method` is not a string
            -`problem_description` is not a string
        ValueError: If any argument has invalid value:
            -`task` not in supported tasks
            -`method` not in supported methods
            -`validation_size` outside [0.0, 1.0]
            -`dataset` is None but `method` requiers a dataset
            -`target` length does not equal the `dataset` length
            -`dataset` is `None` while `target` is not
            -`problem_description` is not provided when using the
                ReflectivePrompt method
    """

    validate_start_prompt(start_prompt)
    task = validate_task(task)
    method = validate_method(method)
    validate_dataset(dataset, target, method)
    validate_target(target, dataset)
    validate_problem_description(problem_description, method)
    validate_validation_size(validation_size)
    return task, method
