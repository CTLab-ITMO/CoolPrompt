from dataclasses import dataclass
from typing import List, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from coolprompt.evaluator.metrics import BaseMetric
from coolprompt.utils.logging_config import logger
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)


@dataclass
class FailedExampleDetailed:
    instance: str
    assistant_answer: str
    model_answer_parsed: Optional[str] = None
    metric_value: float | int = 0.0
    ground_truth: str | int = ""


@dataclass
class EvalResultDetailed:
    aggregate_score: float
    score_per_task: List[float | int] = None
    failed_examples: List[FailedExampleDetailed] = None


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
        dataset: list[str],
        targets: list[str | int],
        template: Optional[str] = None,
    ) -> float:
        """Evaluate the model on a dataset.

        Args:
            prompt (str): The prompt string to prepend to each dataset sample.
            dataset (list[str]): List of input samples to evaluate.
            targets (list[str|int]): Corresponding ground truth labels.
            template (Optional[str]): Prompt template for defined task type.

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
        answers = [
            a.content if isinstance(a, AIMessage) else a for a in answers
        ]

        return self.metric.compute(answers, targets, dataset)

    def evaluate_detailed(
        self,
        prompt: str,
        dataset: list[str],
        targets: list[str | int],
        template: Optional[str] = None,
    ) -> EvalResultDetailed:
        """Evaluate the model and return detailed results per sample."""
        if template is None:
            template = self._get_default_template()

        logger.info(
            f"Evaluating (detailed) prompt for {self.task} task on {len(dataset)} samples"
        )
        if self.task == Task.CLASSIFICATION:
            self.metric.extract_labels(targets)

        answers = self.model.batch(
            [
                self._get_full_prompt(prompt, sample, template)
                for sample in dataset
            ]
        )
        answers = [
            a.content if isinstance(a, AIMessage) else a for a in answers
        ]

        parsed_answers = [self.metric.parse_output(a) for a in answers]
        aggregate_score, score_per_task = self.metric.compute_detailed(
            answers, targets
        )

        failed_examples = []
        for i, score in enumerate(score_per_task):
            if score == 0:
                failed_examples.append(
                    FailedExampleDetailed(
                        instance=dataset[i],
                        assistant_answer=answers[i],
                        model_answer_parsed=parsed_answers[i],
                        metric_value=score,
                        ground_truth=targets[i],
                    )
                )

        return EvalResultDetailed(
            aggregate_score=aggregate_score,
            score_per_task=score_per_task,
            failed_examples=failed_examples,
        )

    def _get_full_prompt(
        self,
        prompt: str,
        sample: str,
        template: Optional[str] = None,
    ) -> str:
        """Inserts parts of the prompt into the task template."""
        if template is None:
            template = self._get_default_template()

        match self.task:
            case Task.CLASSIFICATION:
                labels = ", ".join(map(str, self.metric.label_to_id.keys()))
                return template.format(
                    PROMPT=prompt, LABELS=labels, INPUT=sample
                )
            case Task.GENERATION:
                return template.format(PROMPT=prompt, INPUT=sample)

    def _get_default_template(self) -> str:
        """Returns the default template for the task type."""
        match self.task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE
