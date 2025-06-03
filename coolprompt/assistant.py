from typing import Iterable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.naive import naive_optimizer
from coolprompt.evaluator import Evaluator
from coolprompt.evaluator import CLASSIFICATION_METRICS, GENERATION_METRICS


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    TASK_TYPES = {"classification", "generation"}

    def __init__(self, model: BaseLanguageModel = None) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: BaseLanguageModel - Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str.
                Will use DefaultLLM if not provided.
        """
        self._model = model or DefaultLLM.init()
        self._validate_model()

    def run(
        self,
        start_prompt: str,
        task: str = "generation",
        dataset: Iterable[str] = None,
        target: Iterable[str] | Iterable[int] = None,
        method: str = None,
        metric: str = None,
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
            metric (str): Metric to use for optimization.

        Returns:
            final_prompt: str - The resulting optimized prompt
            after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.

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

        if dataset is None or method is None:
            final_prompt = naive_optimizer(self._model, start_prompt)

        if dataset is not None:
            metric = self._validate_metric(task, metric)
            evaluator = Evaluator(self._model, metric)

            self.init_metric = evaluator.evaluate(
                start_prompt, dataset, target, task)
            self.final_metric = evaluator.evaluate(
                final_prompt, dataset, target, task)

        return final_prompt

    def _validate_metric(self, task: str, metric: str | None) -> str:
        if task not in self.TASK_TYPES:
            raise ValueError(
                f"Invalid task type: {task}. Must be one of {self.TASK_TYPES}."
                )
        if metric is None:
            return self._get_default_metric(task)
        if task == "classification" and metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Invalid metric for {task} task: {metric}."
                f"Must be one of {CLASSIFICATION_METRICS}."
            )
        if task == "generation" and metric not in GENERATION_METRICS:
            raise ValueError(
                f"Invalid metric for {task} task: {metric}."
                f"Must be one of {GENERATION_METRICS}."
            )
        return metric

    def _get_default_metric(self, task: str) -> str:
        if task == "classification":
            return "f1"
        elif task == "generation":
            return "meteor"

    def _validate_model(self) -> None:
        if not isinstance(self._model, BaseLanguageModel):
            raise TypeError(
                "Model should be instance of LangChain BaseLanguageModel"
            )
