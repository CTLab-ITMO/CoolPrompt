from langchain_core.language_models.base import BaseLanguageModel
from typing import Optional

from coolprompt.evaluator.metrics import create_metric
from coolprompt.utils.logging_config import logger
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

    def __init__(self, model: BaseLanguageModel, metric: str) -> None:
        self.model = model
        self.metric = create_metric(metric)
        logger.info(f"Evaluator sucessfully initialized with {metric} metric")

    def evaluate(
        self,
        prompt: str,
        dataset: list[str],
        targets: list[str | int],
        task: str,
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
            task (str):
                The type of task, either "classification" or "generation".
            template (Optional[str]):
                Prompt template for defined task type. If None, uses default template.

        Returns:
            float: The computed evaluation metric score.
        """
        if template is None and task == "classification":
            template = CLASSIFICATION_TASK_TEMPLATE
        elif template is None and task == "generation":
            template = GENERATION_TASK_TEMPLATE

        logger.info(
            f"Evaluating prompt for {task} task on {len(dataset)} samples"
        )
        logger.debug(f"Prompt to evaluate:\n{prompt}")
        if task == "classification":
            self.metric.extract_labels(targets)
        
        answers = self.model.batch(
            [
                self._get_full_prompt(prompt, sample, task, template)
                for sample in dataset
            ]
        )
        return self.metric.compute(answers, targets)

    def _get_full_prompt(
        self, prompt: str, sample: str, task: str, template: str
    ) -> str:
        """Inserts parts of the prompt into the task template.

        Args:
            prompt (str): the main instruction for the task
            sample (str): the input sample
            task (str):
                The type of task, either "classification" or "generation".
            template (str): Prompt template for defined task type

        Raises:
            ValueError: if type of task is not supported

        Returns:
            str: the full prompt to be passed to the model
        """

        if task == "classification":
            labels = ", ".join(map(str, self.metric.label_to_id.keys()))
            return template.format(PROMPT=prompt, LABELS=labels, INPUT=sample)
        elif task == "generation":
            return template.format(PROMPT=prompt, INPUT=sample)
        else:
            error_msg = (
                f"Unknown task type: {task}. "
                f"Available tasks: classification, generation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
