from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator.metrics import create_metric


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
        dataset: list[str],
        targets: list[str | int],
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

        Returns:
            float: The computed evaluation metric score.
        """

        answers = self.model.batch(
            [self._get_full_prompt(prompt, sample) for sample in dataset]
        )
        return self.metric.compute(answers, targets)

    def _get_full_prompt(self, prompt: str, sample: str) -> str:
        if "{<INPUT>}" in prompt:
            return prompt.replace("{<INPUT>}", sample)
        return prompt + "\n" + sample
