from typing import Iterable
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.evaluator import Evaluator, validate_metric
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.naive import naive_optimizer
from coolprompt.optimizer.reflective_prompt import reflectiveprompt
from coolprompt.utils.validation import validate_model
from coolprompt.utils.prompt_template import (CLASSIFICATION_TASK_TEMPLATE,
                                              GENERATION_TASK_TEMPLATE)


class PromptTuner:
    """Prompt optimization tool supporting multiple methods.

        Attributes:
            METHODS: available methods of prompt tuning.
    """

    METHODS = ['naive', 'reflective']

    def __init__(self, model: BaseLanguageModel = None) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: BaseLanguageModel - Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str.
                Will use DefaultLLM if not provided.
        """
        self._model = model or DefaultLLM.init()
        validate_model(self._model)

    def run(
        self,
        start_prompt: str,
        task: str = "generation",
        dataset: Iterable[str] = None,
        target: Iterable[str] | Iterable[int] = None,
        method: str = "naive",
        metric: str = None,
        problem_description: str = None,
    ) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt (str): Initial prompt text to optimize.
            task (str):
                Type of task to optimize for (classification or generation).
                Defaults to generation.
            dataset (Iterable):
                Optional iterable object for dataset-based optimization.
            target (Iterable):
                Target iterable object for dataset-based optimization.
            method (str): Optimization method to use.
                Available methods are: naive and reflective.
                Defaults to naive.
            metric (str): Metric to use for optimization.
            problem_description (str): a string that contains
                short description of problem to optimize.

        Returns:
            final_prompt: str - The resulting optimized prompt
            after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.
            ValueError: If a problem description is not provided
                for ReflectivePrompt.

        Note:
            Uses naive optimization
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
                f"Unsupported method {method}.\n" +
                f"Available methods: {' '.join(self.METHODS)}"
            )

        if method == 'naive':
            final_prompt = naive_optimizer(self._model, start_prompt)

        if dataset is not None:
            metric = validate_metric(task, metric)
            evaluator = Evaluator(self._model, metric)

            if method == 'reflective':
                if problem_description is None:
                    raise ValueError(
                        "Problem description should be provided for " +
                        "ReflectivePrompt optimization"
                    )
                dataset_split = train_test_split(
                    dataset,
                    target,
                    test_size=0.25
                )
                final_prompt = reflectiveprompt(
                    model=self._model,
                    dataset_split=dataset_split,
                    evaluator=evaluator,
                    task=task,
                    problem_description=problem_description,
                    initial_prompt=start_prompt
                )

            self.init_metric = evaluator.evaluate(
                start_prompt, dataset, target, task
            )
            self.final_metric = evaluator.evaluate(
                final_prompt, dataset, target, task
            )

        return final_prompt

    def get_task_prompt_template(task: str) -> str:
        """Returns the prompt template for the given task.

        Args:
            task (str):
                The type of task, either "classification" or "generation".

        Returns:
            str: The prompt template for the given task.
        """

        if task == "classification":
            return CLASSIFICATION_TASK_TEMPLATE
        elif task == "generation":
            return GENERATION_TASK_TEMPLATE
        else:
            raise ValueError(f"Invalid task type: {task}")
