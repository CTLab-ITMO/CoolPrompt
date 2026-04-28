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
    """Unified interface for auto‑prompting methods.

    This abstract base class defines the contract that all prompt
    optimization methods must implement. Concrete subclasses provide
    specific optimization strategies (e.g., evolutionary search,
    distillation, reflection, compression, etc.).
    """

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
        """Run the prompt optimization process.

        Args:
            model (BaseLanguageModel): The language model to be optimized.
            initial_prompt (str): The starting prompt string.
            dataset_split (Tuple[List[str], List[str], List[str], List[str]] | None):
                A tuple of four lists representing (train_inputs, train_labels,
                test_inputs, test_labels). May be None for data‑free methods.
            evaluator (Evaluator | None): An evaluator object for scoring prompts.
                Required for data‑driven methods; can be None for data‑free methods.
            problem_description (str | None): Natural language description of the
                task. May be used to guide the optimization.
            **kwargs: Additional method‑specific arguments.

        Returns:
            str: The optimized prompt.
        """
        pass

    @abstractmethod
    def is_data_driven(self) -> bool:
        """Indicate whether this method requires a dataset for optimization.

        Returns:
            bool: True if the method needs `dataset_split` and `evaluator`,
                  False if it can operate without data.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the optimization method.

        Returns:
            str: A short, lowercase string identifying the method
                 (e.g., "hype", "distill", "compress", "reflective", "regps").
        """
        pass

    def get_template(self, task: Task) -> str:
        """Return the default prompt template for a given task type.

        Subclasses may override this method to provide task‑specific
        templates tailored to their optimization strategy.

        Args:
            task (Task): The task enum value (CLASSIFICATION or GENERATION).

        Returns:
            str: The corresponding template string.
        """
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE
            