from langchain_core.language_models.base import BaseLanguageModel
from typing import Optional

from coolprompt.evaluator.metrics import BaseMetric
from coolprompt.utils.logging_config import logger
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)


class Evaluator:
    """Evaluator class to perform model evaluation using a specified metric.

    This class ties together a language model and an evaluation metric,
    providing a method to generate model outputs on a dataset and compute
    the corresponding metric score against provided targets.
    """

    def __init__(self,
                 model: BaseLanguageModel,
                 task: Task,
                 metric: BaseMetric) -> None:
        self.model = model
        self.task = task
        self.metric = metric
        logger.info(f"Evaluator sucessfully initialized with {metric} metric")

    def evaluate(
        self,
        prompt: str,
        dataset: list[str],
        targets: list[str | int],
        template: Optional[str] = None,
    ) -> float:
        """
        Evaluate the model on a dataset
        by generating answers and computing the metric.

        For each sample in the dataset,
        the prompt is concatenated with the sample,
        passed to the model to generate an output,
        and then all outputs are evaluated
        against the targets using the metric.

        Args:
            prompt (str): The prompt string to prepend to each dataset sample.
            dataset (list[str]): List of input samples to evaluate.
            targets (list[str|int]):
                Corresponding ground truth labels or references.
            template (Optional[str]):
                Prompt template for defined task type.
                If None, uses default template.

        Returns:
            float: The computed evaluation metric score.
        """

        if template is None:
            template = self._get_default_template()

        logger.info(
            f"Evaluating prompt for {self.task} task on {len(dataset)} samples"
        )
        logger.debug(f"Prompt to evaluate:\n{prompt}")
        if self.task == Task.CLASSIFICATION:
            self.metric.extract_labels(targets)

        answers = self.model.batch(
            [
                self._get_full_prompt(prompt, sample, template)
                for sample in dataset
            ]
        )
        return self.metric.compute(answers, targets)

    def _get_full_prompt(
        self, prompt: str, sample: str, template: Optional[str] = None,
    ) -> str:
        """Inserts parts of the prompt into the task template.

        Args:
            prompt (str): the main instruction for the task
            sample (str): the input sample
            template (Optional[str]):
                Prompt template for defined task type.
                If None, uses default template.

        Raises:
            ValueError: if type of task is not supported

        Returns:
            str: the full prompt to be passed to the model
        """

        if template is None:
            template = self._get_default_template()

        match self.task:
            case Task.CLASSIFICATION:
                labels = ", ".join(map(str, self.metric.label_to_id.keys()))
                return template.format(
                    PROMPT=prompt, LABELS=labels, INPUT=sample)
            case Task.GENERATION:
                return template.format(PROMPT=prompt, INPUT=sample)

    def _get_default_template(self) -> str:
        """Returns the default template for the task type."""

        match self.task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE
