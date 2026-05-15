from langchain_core.language_models.base import BaseLanguageModel
from typing import Optional, Tuple, List, Dict, Sequence

from langchain_core.messages.ai import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage
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

    def __init__(
        self, model: BaseLanguageModel, task: Task, metric: BaseMetric
    ) -> None:
        self.model = model
        self.task = task
        self.metric = metric
        logger.info(f"Evaluator successfully initialized with {metric} metric")

    def evaluate(
        self,
        prompt: str,
        dataset: Sequence[str],
        targets: Sequence[str | int],
        template: Optional[str] = None,
        system_role: Optional[str] = None,
        constraints: Optional[str] = None,
        failed_examples: Optional[int] = None,
    ) -> float | Tuple[float, List[Dict]]:
        """Evaluates the prompt on the given dataset and returns the metric score.

        Args:
            prompt (str): the main task description to evaluate.
            dataset (list[str]): input samples to run the model on.
            targets (list[str|int]): ground truth labels or answers.
            template (Optional[str]): prompt template override. Defaults to None.
            system_role (Optional[str]): system behavior / role for the model. Defaults to None.
            constraints (Optional[str]): output format constraints appended to the prompt. Defaults to None.
            failed_examples (Optional[int]): if set, also returns the N worst examples. Defaults to None.

        Returns:
            float | Tuple[float, List[Dict]]: metric score, or (score, bad_examples) if failed_examples is set.
        """
        if template is None:
            template = self._get_default_template()

        logger.info(
            f"Evaluating prompt for {self.task} task on {len(dataset)} samples"
        )
        if system_role:
            logger.debug(
                f"System behavior (system_behavior):\n{system_role}\n"
                f"Task description (task_description):\n{prompt}"
            )
        else:
            logger.debug(f"Task description (task_description):\n{prompt}")
        if constraints:
            logger.debug(f"Output constraints:\n{constraints}")
        if self.task == Task.CLASSIFICATION:
            self.metric.extract_labels(targets)

        answers = self.model.batch(
            [
                self._get_full_prompt(
                    prompt, sample, template, system_role, constraints
                )
                for sample in dataset
            ]
        )
        answers = [
            a.content if isinstance(a, AIMessage) else a for a in answers
        ]

        return self.metric.compute(
            answers, targets, dataset, failed_examples=failed_examples
        )

    def _get_full_prompt(
        self,
        prompt: str,
        sample: str,
        template: Optional[str] = None,
        system_role: Optional[str] = None,
        constraints: Optional[str] = None,
    ) -> str | list:
        """Inserts parts of the prompt into the task template.

        Args:
            prompt (str): the main instruction for the task.
            sample (str): the input sample.
            template (Optional[str]):
                prompt template for the defined task type.
                If None, uses the default template.
            system_role (Optional[str]): system behavior prepended as a SystemMessage. Defaults to None.
            constraints (Optional[str]): output format constraints appended to the prompt. Defaults to None.

        Raises:
            ValueError: if type of task is not supported.

        Returns:
            str | list: the full prompt string, or a list of SystemMessage + HumanMessage if system_role is set.
        """
        if template is None:
            template = self._get_default_template()

        effective_prompt = prompt
        if constraints:
            effective_prompt = f"{prompt}\n\n{constraints}"

        match self.task:
            case Task.CLASSIFICATION:
                labels = ", ".join(map(str, self.metric.label_to_id.keys()))
                formatted_prompt = template.format(
                    PROMPT=effective_prompt, LABELS=labels, INPUT=sample
                )
            case Task.GENERATION:
                formatted_prompt = template.format(
                    PROMPT=effective_prompt, INPUT=sample
                )

        if system_role:
            return [
                SystemMessage(content=system_role),
                HumanMessage(content=formatted_prompt),
            ]
        return formatted_prompt

    def _get_default_template(self) -> str:
        """Returns the default template for the task type."""

        match self.task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE
