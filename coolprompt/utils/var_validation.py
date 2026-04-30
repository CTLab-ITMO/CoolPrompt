from typing import Any, Iterable
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.logging_config import logger
from coolprompt.utils.enums import Task, PD_Method
from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.registry import METHOD_REGISTRY


def validate_verbose(verbose: int) -> None:
    """Validate that the verbose parameter is 0, 1, or 2.

    Args:
        verbose (int): Verbosity level to validate.

    Raises:
        ValueError: If `verbose` is not 0, 1, or 2.
    """
    if verbose not in [0, 1, 2]:
        error_msg = f"Invalid verbose: {verbose}. Available values: 0, 1, 2."
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_model(model: BaseLanguageModel) -> None:
    """Validate that the model is a LangChain BaseLanguageModel instance.

    Args:
        model (BaseLanguageModel): Model instance to validate.

    Raises:
        TypeError: If `model` is not an instance of BaseLanguageModel.
    """
    if not isinstance(model, BaseLanguageModel):
        error_msg = (
            "Provided model must be an "
            "instance of LangChain BaseLanguageModel"
        )
        logger.error(error_msg)
        raise TypeError(error_msg)


def validate_start_prompt(start_prompt: str) -> None:
    """Validate that the start prompt is a non‑empty string.

    Args:
        start_prompt (str): Initial prompt to validate.

    Raises:
        TypeError: If `start_prompt` is not a string or is empty.
    """
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
    """Validate the task type and return the corresponding Task enum.

    Args:
        task (str): Task type, must be "classification" or "generation".

    Returns:
        Task: The validated Task enum member.

    Raises:
        TypeError: If `task` is not a string.
        ValueError: If `task` is not a known task name.
    """
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
        error_msg = f"Invalid task type: {task}. " f"Available tasks: {
            ', '.join(
                list(
                    Task._value2member_map_.keys()))}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    return Task(task)


def validate_dataset(
    dataset: Iterable | None,
    target: Iterable | None,
    method: AutoPromptingMethod,
) -> None:
    """Validate dataset and target consistency for a given method.

    Args:
        dataset (Iterable | None): Input dataset to validate.
        target (Iterable | None): Target labels corresponding to the dataset.
        method (AutoPromptingMethod): Optimization method (used to check
            data‑driven requirements).

    Raises:
        TypeError: If `dataset` is not None but not an Iterable.
        ValueError: If `dataset` is None while `method` requires a dataset,
            or if `dataset` is provided but `target` is None,
            or if `dataset` is empty and the method expects data.
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
                    f"optimization method '{
                        method.name}'. You can try using HyPE "
                    "optimization ('hype' as method parameter) which "
                    "does not require any train dataset."
                )
            else:
                error_msg = (
                    "Dataset must be non-empty for evaluation when using "
                    f"'{method.name}' optimization method. If you do not want to "
                    "evaluate your prompts, please do not provide any dataset."
                )
            logger.error(error_msg)
            raise ValueError(error_msg)


def validate_target(target: Iterable | None, dataset: Iterable | None) -> None:
    """Validate that the target is an Iterable and matches dataset length.

    Args:
        target (Iterable | None): Target labels to validate.
        dataset (Iterable | None): Dataset that must be provided if target is given.

    Raises:
        TypeError: If `target` is not None but not Iterable.
        ValueError: If `target` is not None and `dataset` is None,
            or if lengths of `target` and `dataset` differ.
    """
    if target is not None:
        if dataset is None:
            error_msg = "Dataset cannot be None if target is provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(target, Iterable):
            error_msg = (
                "Target must be an Iterable instance. "
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


def validate_method(method: str | AutoPromptingMethod) -> AutoPromptingMethod:
    """Validate and return the AutoPromptingMethod instance.

    Args:
        method (str | AutoPromptingMethod): Method name (string) or an instance.

    Returns:
        AutoPromptingMethod: The validated method instance.

    Raises:
        TypeError: If `method` is not a string.
        ValueError: If `method` is not one of
            ["hype", "hyper", "reflective", "distill", "compress"].
    """

    if not isinstance(method, str):
        error_msg = (
            "Method must be a string or AutoPromptingMethod instance. "
            f"Provided: {type(method).__name__}."
        )
        logger.error(error_msg)
        raise TypeError(error_msg)
    return method_impl


def validate_validation_size(validation_size: float | Any) -> None:
    """Validate that validation_size is a float between 0.0 and 1.0.

    Args:
        validation_size (float | Any): Value to validate.

    Raises:
        ValueError: If `validation_size` is not a float in [0.0, 1.0].
    """
    if not isinstance(validation_size, float) or not (
        0.0 <= validation_size <= 1.0
    ):
        error_msg = (
            "Validation size must be a float between 0.0 and 1.0. "
            f"Provided: {validation_size}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_problem_description(
    problem_description: str | None, pd_method: str
) -> PD_Method:
    """Validate problem description and its generation method.

    Args:
        problem_description (str | None): Optional problem description text.
        pd_method (str): Problem description generation method,
            must be one of ["base", "dataset-based"].

    Returns:
        PD_Method: Validated PD_Method enum member.

    Raises:
        TypeError: If `problem_description` is not None but not a string.
        ValueError: If `pd_method` is not a known method name.
    """
    if problem_description is not None:
        if not isinstance(problem_description, str):
            error_msg = (
                "Problem description must be a string. "
                f"Provided: {type(problem_description).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

    if pd_method not in PD_Method._value2member_map_:
        error_msg = (
            f"Unknown problem description generation method: {pd_method}. "
            f"Available methods: {list(PD_Method._value2member_map_.keys())}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    pd_method_impl = PD_Method(pd_method)
    return pd_method_impl


def validate_run(
    start_prompt: str,
    task: str,
    dataset: Iterable | None,
    target: Iterable | None,
    method: str | AutoPromptingMethod,
    problem_description: str | None,
    problem_description_generation_method: str,
    validation_size: float,
) -> tuple[Task, AutoPromptingMethod, PD_Method]:
    """Validate all arguments for PromptTuner.run().

    Args:
        start_prompt (str): Initial prompt string (must be non‑empty).
        task (str): Task type, one of "classification" or "generation".
        dataset (Iterable | None): Input dataset. Required for data‑driven methods.
        target (Iterable | None): Target labels matching the dataset.
        method (str | AutoPromptingMethod): Optimization method name or instance.
        problem_description (str | None): Task description (may be required by some methods).
        problem_description_generation_method (str): Method to generate problem description,
            one of "base" or "dataset-based".
        validation_size (float): Fraction of dataset to use for validation (0.0‑1.0).

    Returns:
        tuple[Task, AutoPromptingMethod, PD_Method]:
            - Validated Task enum
            - Validated AutoPromptingMethod instance
            - Validated PD_Method enum

    Raises:
        TypeError: For incorrect argument types (string, Iterable, etc.).
        ValueError: For invalid values (unknown task/method, size out of range,
            missing dataset for data‑driven method, length mismatch, etc.).
    """
    method_impl = validate_method(method)
    validate_start_prompt(start_prompt)
    task_value = validate_task(task)
    validate_dataset(dataset, target, method_impl)
    validate_target(target, dataset)
    validate_validation_size(validation_size)
    pd_method = validate_problem_description(
        problem_description, problem_description_generation_method
    )
    return task_value, method_impl, pd_method
