import pandas as pd
from language_model.llm import DefaultLLM
from langchain_core.language_models.base import BaseLanguageModel
from utils import NAIVE_AUTOPROMPTING_PROMPT_TEMPLATE


class PromptHelper:
    """Docstring"""

    def __init__(self, model: BaseLanguageModel = None):
        """Docstring

        Args:
            model: langchain_core.language_models.base.BaseLanguageModel
            #config: dict
            #token: str = None
            #interface: str - [hf, ollama, openapi]
        """
        # token = ...
        # config = ...
        self._model = model if model is not None else DefaultLLM.create()

    def invoke(self, start_prompt: str, dataset: pd.DataFrame = None, target: str = None) -> str:
        """Docstring

        Args:
            start_prompt: str
            dataset: pd.DataFrame
            target: str
        Returns:
            final_prompt: str
        """
        if dataset is None:
            return self._naive_autoprompting(start_prompt)

    def _naive_autoprompting(self, prompt: str) -> str:
        template = NAIVE_AUTOPROMPTING_PROMPT_TEMPLATE
        answer = self._model.invoke(template.replace("<PROMPT>", prompt)).strip()
        return answer[answer.find("Rewritten prompt:\n") + len("Rewritten prompt:\n") :]
