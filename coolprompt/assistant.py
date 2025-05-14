import pandas as pd
from coolprompt.language_model.llm import DefaultLLM
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.optimizer.naive import naive_optimizer


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    def __init__(self, model: BaseLanguageModel = None):
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            model: Any LangChain BaseLanguageModel instance. Will use DefaultLLM if not provided.
        """
        self._model = model if model is not None else DefaultLLM.init({"max_model_len": 1600})

    def run(self, start_prompt: str, dataset: pd.DataFrame = None, target: str = None, method: str = None) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt: Initial prompt text to optimize.
            dataset: Optional DataFrame for dataset-based optimization.
            target: Target column name for dataset-based optimization.
            method: Optimization method to use.
        Returns:
            final_prompt (str): The resulting optimized prompt after applying the selected method.
        Note:
            Uses naive optimization when dataset or method parameters are not provided.
        """
        if dataset is None or method is None:
            return naive_optimizer(self._model, start_prompt)
