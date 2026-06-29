from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
from time import sleep
from dataclasses import dataclass

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
import numpy as np
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
    raw_outputs: List[str] = None


class Evaluator:
    """Evaluator class to perform model evaluation using a specified metric.

    This class ties together a language model and an evaluation metric,
    providing a method to generate model outputs on a dataset and compute
    the corresponding metric score against provided targets.
    """

    def __init__(
            self, model: BaseLanguageModel, task: Task, metric: BaseMetric, batch_size: int = 25
    ) -> None:
        self.model = model
        self.task = task
        self.metric = metric
        self.batch_size = batch_size
        logger.info(f"Evaluator successfully initialized with {metric} metric")

    def evaluate(
        self,
        prompt: str,
        dataset: list[str],
        targets: list[str | int],
        template: Optional[str] = None,
        failed_examples: Optional[int] = None,
        *,
        return_detailed: bool = False,
    ) -> float | Tuple[float, List[Dict[str, str]]] | EvalResultDetailed:
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
            failed_examples (Optional[int]):
                Number of bad examples to return after evaluating
            return_detailed (bool, default=False): If True, returns EvalResultDetailed with per-task scores
                and raw outputs.


        Returns:
            float | Tuple[float, List[Dict[str, str]]]:
                The computed evaluation metric score with/wo bad examples, or detailed results.
        """

        if template is None:
            template = self._get_default_template()

        logger.info(
            f"Evaluating prompt for {self.task} task on {len(dataset)} samples"
        )
        if self.task == Task.CLASSIFICATION:
            self.metric.extract_labels(targets)
        full_prompts = [
            self._get_full_prompt(prompt, sample, template)
            for sample in dataset
        ]

        answers = self._run_batches(full_prompts)

        if not return_detailed:
            return self.metric.compute(answers, targets, dataset, failed_examples)

        aggregate, score_per_task, _ = self.metric.compute(
            answers, targets, dataset, failed_examples, return_per_task=True
        )
        parsed_answers = [self.metric.parse_output(a) for a in answers]

        detailed_failures = []
        if failed_examples and failed_examples > 0:
            bad_indices = [i for i, s in enumerate(score_per_task) if s < 1.0]
            bad_indices.sort(key=lambda i: score_per_task[i])
            for i in bad_indices[:failed_examples]:
                detailed_failures.append(
                    FailedExampleDetailed(
                        instance=dataset[i],
                        assistant_answer=answers[i],
                        model_answer_parsed=parsed_answers[i],
                        metric_value=score_per_task[i],
                        ground_truth=targets[i],
                    )
                )

        return EvalResultDetailed(
            aggregate_score=aggregate,
            score_per_task=score_per_task,
            failed_examples=detailed_failures,
            raw_outputs=answers,
        )

    def _run_batches(self, full_prompts: list[str]) -> list[str]:
        """Run the model on preformatted prompts in batches with progress tracking."""
        answers: list[str] = []
        total = len(full_prompts)
        total_batches = (total + self.batch_size - 1) // self.batch_size

        with tqdm(
                total=total,
                desc="Evaluating",
                unit="sample",
                dynamic_ncols=True,
        ) as pbar:
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                batch = full_prompts[start:end]

                batch_answers = None
                for attempt in range(5):
                    try:
                        batch_answers = self.model.batch(batch)
                        break
                    except Exception as exception:
                        logger.warning(
                            f"Batch {start // self.batch_size + 1}/{total_batches} "
                            f"failed on attempt {attempt + 1}/5: {exception}"
                        )
                        if attempt < 4:
                            sleep(60)
                        else:
                            raise RuntimeError(
                                f"Batch {start // self.batch_size + 1}/{total_batches} failed after 5 attempts"
                            ) from exception

                normalized_answers = [
                    a.content if isinstance(a, AIMessage) else str(a)
                    for a in batch_answers
                ]
                answers.extend(normalized_answers)
                pbar.update(len(batch))
                logger.debug(f"Batch {start // self.batch_size + 1}/{total_batches} processed")
        return answers

    def _get_full_prompt(
            self,
            prompt: str,
            sample: str,
            template: Optional[str] = None,
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
