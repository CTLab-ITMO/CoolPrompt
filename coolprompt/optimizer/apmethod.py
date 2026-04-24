from abc import ABC, abstractmethod
from typing import Tuple, List
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator


class AutoPromptingMethod(ABC):
    """Unified interface for autoprompting methods."""

    @abstractmethod
    def optimize(
        self,
        model: BaseLanguageModel,
        initial_prompt: str,
        dataset_split: Tuple[List[str], List[str], List[str], List[str]] | None,
        evaluator: Evaluator | None,
        problem_description: str | None,
        **kwargs,
    ) -> str:
        pass