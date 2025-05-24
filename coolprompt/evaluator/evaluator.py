from typing import List
from langchain_core.language_models.base import BaseLanguageModel
from metrics import BaseMetric


class Evaluator():
    """Evaluator class to perform model evaluation using a specified metric.

    This class ties together a language model and an evaluation metric,
    providing a method to generate model outputs on a dataset and compute
    the corresponding metric score against provided targets.
    """
    
    def __init__(self, model: BaseLanguageModel, metric: BaseMetric) -> None:
        self.model = model
        self.metric = metric

    def evaluate(
        self,
        prompt: str,
        dataset: List[str],
        targets: List[str] | List[int],
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

        Returns:
            float: The computed evaluation metric score.
        """
        
        answers = [self.model.invoke(prompt + "\n" + sample) for sample in dataset]
        return self.metric.compute(answers, targets)
