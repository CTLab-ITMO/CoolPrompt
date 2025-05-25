import pandas as pd
from langchain_core.language_models.base import BaseLanguageModel
from typing import Iterable

from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.naive import naive_optimizer
from coolprompt.evaluator.metrics import BaseMetric
from coolprompt.evaluator.evaluator import Evaluator


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    TASK_TYPES = {"classification", "generation"}

    def __init__(self, model: BaseLanguageModel = None) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: BaseLanguageModel - Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str. Will use DefaultLLM if not provided.
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
            task (str): Type of task to optimize for (classification or generation).
            dataset (Iterable): Optional iterable object for dataset-based optimization.
            target (Iterable): Target iterable object for dataset-based optimization.
            method (str): Optimization method to use.
            metric (str): Metric to use for optimization.
            
        Returns:
            final_prompt: str - The resulting optimized prompt after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.
            
        Note:
            Uses naive optimization when dataset or method parameters are not provided.
            
            Uses default metric for the task type if metric parameter is not provided:
            f1 for classisfication, meteor for generation.
        """

        if metric is None:
            metric = self._get_default_metric(task)

        evaluator = Evaluator(self._model, metric)
        
        if dataset is not None and metric is not None:
            self.init_metric = evaluator.evaluate(start_prompt, dataset, target)
        
        final_prompt = ""
        if dataset is None or method is None:
            final_prompt = naive_optimizer(self._model, start_prompt)
        
        if dataset is not None and metric is not None:
            self.final_metric = evaluator.evaluate(final_prompt, dataset, target)
        
        return final_prompt

    def _get_default_metric(task: str) -> str:
        if task == "classification":
            return "f1"
        elif task == "generation":
            return "meteor"
        else:
            raise ValueError(f"Invalid task type: {task}. Must be one of {PromptTuner.TASK_TYPES}.")

    def _validate_model(self):
        if not isinstance(self._model, BaseLanguageModel):
            raise TypeError("Model should be instance of LangChain BaseLanguageModel")
        if not hasattr(self._model, "invoke"):
            raise AttributeError("Model should implement .invoke()")
