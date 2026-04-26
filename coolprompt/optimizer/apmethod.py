from abc import ABC, abstractmethod
from typing import Tuple, List
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.enums import Task


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

    @abstractmethod
    def is_data_driven(self) -> bool:        
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def get_template(self, task: Task) -> str:
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE
