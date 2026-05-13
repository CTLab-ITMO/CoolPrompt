from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.enums import Task
from coolprompt.utils.load_dataset import load_dataset
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.utils import get_dataset_split


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


def _parse_dataset_size(size: str) -> Optional[int]:
    """Return integer dataset size, or None if ``all``."""

    if size == "all":
        return None
    return int(size)


class ConfiguredAutoPromptingMethod:
    """YAML/config harness: load train/val/test, run ``_run``, score on val and test."""

    def __init__(
        self, model: BaseLanguageModel, config: Dict[str, Any]
    ) -> None:
        from coolprompt.utils.var_validation import validate_task

        self.model = model
        self.config = config
        self._system_model = model

        data_split = self.config["dataset"]["configuration"]
        data_split = data_split.split("/")
        train_size = _parse_dataset_size(data_split[0])
        val_size = _parse_dataset_size(data_split[1])
        test_size = _parse_dataset_size(data_split[2])

        train_dataset, train_target = load_dataset(
            self.config["dataset"]["name"],
            size=train_size + val_size,
            split="train",
        )

        self.dataset_split = get_dataset_split(
            dataset=train_dataset,
            target=train_target,
            validation_size=val_size / (train_size + val_size),
            train_as_test=self.config.get("train_as_test", False),
        )

        self.test_dataset, self.test_target = load_dataset(
            self.config["dataset"]["name"], size=test_size, split="test"
        )

        task = validate_task(self.config["task"])
        metric = validate_and_create_metric(task, self.config["metric"])
        self.evaluator = Evaluator(self.model, task, metric)

    def _run(self, start_prompt: str) -> str:
        raise NotImplementedError

    def run(
        self, start_prompt: str, saving_model_answers: bool = False
    ) -> None:
        self.final_prompt = self._run(start_prompt)

        self.final_val_score = self.evaluator.evaluate(
            prompt=self.final_prompt,
            dataset=self.dataset_split[1],
            targets=self.dataset_split[3],
        )

        self.final_test_score = self.evaluator.evaluate(
            prompt=self.final_prompt,
            dataset=self.test_dataset,
            targets=self.test_target,
            save_model_answers=saving_model_answers,
            model_answers_output_path=self.config.get(
                "model_answers_output_path", "./model_answers.yaml"
            ),
        )
