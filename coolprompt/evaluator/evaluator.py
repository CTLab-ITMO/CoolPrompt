from typing import List
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.evaluator.metrics import InputType
from coolprompt.evaluator.metrics import create_metric
from coolprompt.utils.prompt_template import CLASSIFICATION_TASK_TEMPLATE, GENERATION_TASK_TEMPLATE


class Evaluator():
    """Evaluator class to perform model evaluation using a specified metric.

    This class ties together a language model and an evaluation metric,
    providing a method to generate model outputs on a dataset and compute
    the corresponding metric score against provided targets.
    """
    
    def __init__(self, model: BaseLanguageModel, metric: str) -> None:
        self.model = model
        self.metric = create_metric(metric)

    def evaluate(
        self,
        prompt: str,
        dataset: List[str],
        targets: InputType,
        task: str,
    ) -> float:
        """
        Evaluate the model on a dataset by generating answers and computing the metric.

        For each sample in the dataset, the prompt is concatenated with the sample,
        passed to the model to generate an output, and then all outputs are evaluated
        against the targets using the metric.

        Args:
            prompt (str): The prompt string to prepend to each dataset sample.
            dataset (List[str]): List of input samples to evaluate.
            targets (List[str] | List[int]): Corresponding ground truth labels or references.
            task (str): The type of task, either "classification" or "generation".

        Returns:
            float: The computed evaluation metric score.
        """

        if task == "classification":
            self.metric.extract_labels(targets)
        for sample in dataset[:10]:
            print(self._get_full_prompt(prompt, sample, task))
        answers = self.model.batch([self._get_full_prompt(prompt, sample, task) for sample in dataset])
        return self.metric.compute(answers, targets)

    def _get_full_prompt(self, prompt: str, sample: str, task: str) -> str:
        if task == "classification":
            labels = ', '.join(map(str, self.metric.label_to_id.keys()))
            return CLASSIFICATION_TASK_TEMPLATE.format(PROMPT=prompt, LABELS=labels, INPUT=sample)
        elif task == "generation":
            return GENERATION_TASK_TEMPLATE.format(PROMPT=prompt, INPUT=sample)
        else:
            raise ValueError(f"Unknown task type: {task}")
