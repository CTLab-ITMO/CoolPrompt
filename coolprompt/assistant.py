from pathlib import Path
from typing import Iterable, Optional
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.optimizer.reflective_prompt import reflectiveprompt
from coolprompt.optimizer.distill_prompt.run import distillprompt
from coolprompt.utils.logging_config import logger, set_verbose, setup_logging
from coolprompt.utils.var_validation import (
    validate_model,
    validate_task,
    validate_method,
    validate_run,
    validate_verbose,
)
from coolprompt.utils.enums import Method, Task
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    TEMPLATE_MAP = {
        (Task.CLASSIFICATION, Method.HYPE): CLASSIFICATION_TASK_TEMPLATE_HYPE,
        (Task.CLASSIFICATION, Method.REFLECTIVE): CLASSIFICATION_TASK_TEMPLATE,
        (Task.CLASSIFICATION, Method.DISTILL): CLASSIFICATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.HYPE): GENERATION_TASK_TEMPLATE_HYPE,
        (Task.GENERATION, Method.REFLECTIVE): GENERATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.DISTILL): GENERATION_TASK_TEMPLATE,
    }

    def __init__(
        self,
        model: BaseLanguageModel = None,
        logs_dir: str | Path = None,
    ) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model (BaseLanguageModel): Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str.
                Will use DefaultLLM if not provided.
            logs_dir (str | Path, optional): logs saving directory.
                Defaults to None.
        """
        setup_logging(logs_dir)
        self._model = model or DefaultLLM.init()
        self.init_metric = None
        self.init_prompt = None
        self.final_metric = None
        self.final_prompt = None

        logger.info(f"Validating the model: {self._model.__class__.__name__}")
        validate_model(self._model)
        logger.info(
            "PromptTuner successfully initialized with "
            f"model: {self._model.__class__.__name__}"
        )

    def get_task_prompt_template(self, task: str, method: str) -> str:
        """Returns the prompt template for the given task.

        Args:
            task (str):
                The type of task, either "classification" or "generation".
            method (str):
                Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']

        Returns:
            str: The prompt template for the given task.
        """

        logger.debug(
            f"Getting prompt template for {task} task and {method} method"
        )
        task = validate_task(task)
        method = validate_method(method)
        return self.TEMPLATE_MAP[(task, method)]

    def run(
        self,
        start_prompt: str,
        task: str = "generation",
        dataset: Optional[Iterable[str]] = None,
        target: Optional[Iterable[str] | Iterable[int]] = None,
        method: str = "hype",
        metric: Optional[str] = None,
        problem_description: Optional[str] = None,
        validation_size: float = 0.25,
        verbose: int = 1,
        **kwargs,
    ) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt (str): Initial prompt text to optimize.
            task (str):
                Type of task to optimize for (classification or generation).
                Defaults to generation.
            dataset (Iterable):
                Dataset iterable object for autoprompting optimization.
            target (Iterable):
                Target iterable object for autoprompting optimization.
            method (str): Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']
                Defaults to hype.
            metric (str): Metric to use for optimization.
            problem_description (str): a string that contains
                short description of problem to optimize.
            validation_size (float):
                A float that must be between 0.0 and 1.0 and
                represent the proportion of the dataset
                to include in the validation split.
                Defaults to 0.25.
            verbose (int): Parameter for logging configuration:
                0 - no logging
                1 - steps logging
                2 - steps and prompts logging
            **kwargs (dict[str, Any]): other key-word arguments.

        Returns:
            final_prompt: str - The resulting optimized prompt
            after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.
            ValueError: If a problem description is not provided
                for ReflectivePrompt.

        Note:
            Uses HyPE optimization
            when dataset or method parameters are not provided.

            Uses default metric for the task type
            if metric parameter is not provided:
            f1 for classisfication, meteor for generation.

            if dataset is not None, you can find evaluation results
            in self.init_metric and self.final_metric
        """
        if verbose is not None:
            validate_verbose(verbose)
            set_verbose(verbose)

        logger.info("Validating args for PromptTuner running")
        task, method = validate_run(
            start_prompt,
            task,
            dataset,
            target,
            method,
            problem_description,
            validation_size,
        )
        metric = validate_and_create_metric(task, metric)
        evaluator = Evaluator(self._model, task, metric)
        final_prompt = ""

        logger.info("=== Starting Prompt Optimization ===")
        logger.info(f"Method: {method}, Task: {task}")
        logger.info(f"Metric: {metric}, Validation size: {validation_size}")
        if dataset:
            logger.info(f"Dataset: {len(dataset)} samples")
        else:
            logger.info("No dataset provided")
        if target:
            logger.info(f"Target: {len(target)} samples")
        else:
            logger.info("No target provided")
        if kwargs:
            logger.debug(f"Additional kwargs: {kwargs}")

        if method is Method.HYPE:
            final_prompt = hype_optimizer(self._model, start_prompt)
        elif method is Method.REFLECTIVE:
            dataset_split = train_test_split(
                dataset, target, test_size=validation_size
            )
            final_prompt = reflectiveprompt(
                model=self._model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                problem_description=problem_description,
                initial_prompt=start_prompt,
                **kwargs,
            )
        elif method is Method.DISTILL:
            dataset_split = train_test_split(dataset, target, test_size=0.25)
            final_prompt = distillprompt(
                model=self._model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                initial_prompt=start_prompt,
                **kwargs,
            )

        logger.debug(f"Final prompt:\n{final_prompt}")
        if dataset is not None:
            template = self.TEMPLATE_MAP[(task, method)]
            logger.info(f"Evaluating on given dataset for {task} task...")
            self.init_metric = evaluator.evaluate(
                start_prompt, dataset, target, template
            )
            self.final_metric = evaluator.evaluate(
                final_prompt, dataset, target, template
            )
            logger.info(
                f"Initial {metric} score: {self.init_metric}, "
                f"final {metric} score: {self.final_metric}"
            )

        self.init_prompt = start_prompt
        self.final_prompt = final_prompt

        logger.info("=== Prompt Optimization Completed ===")
        return final_prompt
