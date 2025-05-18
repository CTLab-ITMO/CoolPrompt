import pandas as pd
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.naive import naive_optimizer


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    def __init__(self, model: BaseLanguageModel = None):
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: BaseLanguageModel - Any LangChain BaseLanguageModel instance
                which supports invoke(str) -> str. Will use DefaultLLM if not provided.
        """
        self._model = model or DefaultLLM.init()
        self._validate_model()

    def run(self, start_prompt: str, dataset: pd.DataFrame = None, target: str = None, method: str = None) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt: str - Initial prompt text to optimize.
            dataset: DataFrame - Optional Pandas DataFrame for dataset-based optimization.
            target: str - Target column name for dataset-based optimization.
            method: str - Optimization method to use.
        Returns:
            final_prompt: str - The resulting optimized prompt after applying the selected method.
        Note:
            Uses naive optimization when dataset or method parameters are not provided.
        """
        final_prompt = ""
        if dataset is None or method is None:
            final_prompt = naive_optimizer(self._model, start_prompt)
        return final_prompt

    def _validate_model(self):
        if not isinstance(self._model, BaseLanguageModel):
            raise TypeError("Model should be instance of LangChain BaseLanguageModel")
        if not hasattr(self._model, "invoke"):
            raise AttributeError("Model should implement .invoke()")
