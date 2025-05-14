import pandas as pd
from language_model.llm import DefaultLLM
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.optimizer.naive import naive_optimizer


class PromptTuner:
    """Docstring"""

    def __init__(self, model: BaseLanguageModel = None):
        """Docstring

        Args:
            model: langchain_core.language_models.base.BaseLanguageModel
        """
        self._model = model if model is not None else DefaultLLM.init()

    def run(self, start_prompt: str, dataset: pd.DataFrame = None, target: str = None, method: str = None) -> str:
        """Docstring

        Args:
            start_prompt: str
            dataset: pd.DataFrame
            target: str
            method: str
        Returns:
            final_prompt: str
        """
        if dataset is None or method is None:
            return naive_optimizer(start_prompt)
