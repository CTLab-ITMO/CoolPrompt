from typing import Iterable
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.evaluator import Evaluator, validate_metric
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.optimizer.reflective_prompt import reflectiveprompt
from coolprompt.optimizer.distill_prompt.run import distillprompt
from coolprompt.utils.validation import validate_model
from coolprompt.utils.prompt_templates.reflective_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    METHODS = ["hype", "reflective", 'distill']

    TEMPLATE_MAP = {
        ("classification", "hype"): CLASSIFICATION_TASK_TEMPLATE_HYPE,
        ("classification", "reflective"): CLASSIFICATION_TASK_TEMPLATE,
        ("classification", "distill"): CLASSIFICATION_TASK_TEMPLATE,
        ("generation", "hype"): GENERATION_TASK_TEMPLATE_HYPE,
        ("generation", "reflective"): GENERATION_TASK_TEMPLATE,
        ("generation", "distill"): GENERATION_TASK_TEMPLATE,
    }

    def __init__(self, model: BaseLanguageModel = None) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: BaseLanguageModel - Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str.
                Will use DefaultLLM if not provided.
        """
        self._model = model or DefaultLLM.init()
        self.init_metric = None
        self.init_prompt = None
        self.final_metric = None
        self.final_prompt = None

        validate_model(self._model)

    def get_task_prompt_template(self, task: str, method: str) -> str:
        """Returns the prompt template for the given task.

        Args:
            task (str):
                The type of task, either "classification" or "generation".
            method (str):
                Optimization method to use.
                Available methods are: ['hype', 'reflective']

        Returns:
            str: The prompt template for the given task.
        """

        if task not in ["classification", "generation"]:
            raise ValueError(f"Invalid task type: {task}")
        if method not in self.METHODS:
            raise ValueError(f"Invalid method: {method}")
        return self.TEMPLATE_MAP[(task, method)]

    def run(
        self,
        start_prompt: str,
        task: str = "generation",
        dataset: Iterable[str] = None,
        target: Iterable[str] | Iterable[int] = None,
        method: str = "hype",
        metric: str = None,
        problem_description: str = None,
        validation_size: float = 0.25,
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
                A float that should be between 0.0 and 1.0 and
                represent the proportion of the dataset
                to include in the validation split.
                Defaults to 0.25.
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
        final_prompt = ""

        if method not in self.METHODS:
            raise ValueError(
                f"Unsupported method {method}.\n"
                + f"Available methods: {', '.join(self.METHODS)}"
            )

        if dataset is not None:
            if target is None:
                raise ValueError("Must provide target with dataset")
            if len(dataset) != len(target):
                raise ValueError("Dataset and target must have equal length")
            metric = validate_metric(task, metric)
            evaluator = Evaluator(self._model, metric)

        if method == "hype":
            final_prompt = hype_optimizer(self._model, start_prompt)
        elif method == "reflective":
            if problem_description is None:
                raise ValueError(
                    "Problem description should be provided for "
                    "ReflectivePrompt optimization"
                )
            if dataset is None:
                raise ValueError(
                    "Train dataset is not defined for "
                    "ReflectivePrompt optimization"
                )
            dataset_split = train_test_split(
                dataset, target, test_size=validation_size
            )
            final_prompt = reflectiveprompt(
                model=self._model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                task=task,
                problem_description=problem_description,
                initial_prompt=start_prompt,
                **kwargs,
            )
        elif method == 'distill':
            if start_prompt is None:
                raise ValueError(
                    "Starting prompt should be provided for "
                    "DistillPrompt optimization"
                )
            if dataset is None:
                raise ValueError(
                    "Train dataset is not defined for "
                    "DistillPrompt optimization"
                )

            dataset_split = train_test_split(
                dataset,
                target,
                test_size=0.25
            )
            final_prompt = distillprompt(
                model=self._model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                task=task,
                initial_prompt=start_prompt,
                **kwargs,
            )

        if dataset is not None:
            template = self.get_task_prompt_template(task, method)
            self.init_metric = evaluator.evaluate(
                start_prompt, dataset, target, task, template
            )
            self.final_metric = evaluator.evaluate(
                final_prompt, dataset, target, task, template
            )

        self.init_prompt = start_prompt
        self.final_prompt = final_prompt

        return final_prompt
