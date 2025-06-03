from abc import ABC, abstractmethod
from evaluate import load
from coolprompt.utils.parsing import extract_answer


CLASSIFICATION_METRICS = {
    "accuracy",
    "f1",
}

GENERATION_METRICS = {
    "bleu",
    "rouge",
    "meteor",
}


class BaseMetric(ABC):
    """Abstract base class for implementing evaluation metrics.

    Provides common infrastructure for loading metrics
    from HuggingFace's evaluate library and defining
    metric computation interfaces.
    """

    def __init__(self, name: str) -> None:
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        self._name = name
        self._metric = load(name)
        self._compute_kwargs = {}

    def _compute_raw(self,
                     outputs: list[str | int],
                     targets: list[str | int]) -> float:
        """Compute metric value from preprocessed model answers.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            float: Computed metric value
        """

        return self._metric.compute(
            predictions=outputs,
            references=targets,
            **self._compute_kwargs)[self._name]

    @abstractmethod
    def compute(self,
                outputs: list[str | int],
                targets: list[str | int]) -> float:
        """Compute metric value from text model outputs

        Must be implemented by subclasses to handle input formatting.

        Args:
            outputs (list[str|int]): Model predictions (just text)
            targets (list[str|int]): Ground truth labels
        Returns:
            float: Computed metric value
        """
        pass


class ClassificationMetric(BaseMetric):
    """Base class for classification metrics with answer parsing functionality.

    Handles extraction of labels from model outputs
    containing XML-style <ans> tags
    and label encoding for metric computation.

    Attributes:
        ANS_TAGS: tuple - Start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: int - Special value indicating parsing failure
    """

    ANS_TAGS = ("<ans>", "</ans>")
    FORMAT_MISMATCH_LABEL = -1

    def __init__(self, name: str):
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """
        super().__init__(name)
        self.label_to_id = None
        if name == "f1":
            self._compute_kwargs = {"average": "macro"}

    def _encode_labels(
        self,
        output_labels: list[str | int],
        targets: list[str | int]
    ) -> tuple[list[int], list[int]]:
        """Encode string labels into integer IDs for both outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        label_ids = dict()
        encoded_output_labels = []
        encoded_targets = []

        for label in output_labels:
            if label not in label_ids:
                label_ids[label] = len(label_ids)
            encoded_output_labels.append(label_ids[label])

        for label in targets:
            if label not in label_ids:
                label_ids[label] = len(label_ids)
            encoded_targets.append(label_ids[label])

        return encoded_output_labels, encoded_targets

    def compute(self,
                outputs: list[str | int],
                targets: list[str | int]) -> float:
        """Compute the classification metric from
        model outputs and ground truth targets.

        This method extracts labels from outputs,
        encodes them along with targets,
        and computes the metric value.

        Args:
            outputs (list[str|int]): Model output strings.
            targets (list[str|int]): Ground truth labels.
        Returns:
            float: The computed metric value.
        """
        output_labels = list(map(
                lambda x: extract_answer(
                    x,
                    self.ANS_TAGS,
                    self.FORMAT_MISMATCH_LABEL
                ),
                outputs
        ))
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(
            output_labels,
            targets
        )
        return self._compute_raw(encoded_output_labels, encoded_targets)


class GenerationMetric(BaseMetric):
    """Base class for generation metrics.

    Provides a generic implementation for metrics that compare generated text
    to reference text.
    """

    def __init__(self, name: str):
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        super().__init__(name)
        if name == "rouge":
            self._name = "rougeL"

    def compute(self,
                outputs: list[str | int],
                targets: list[str | int]) -> float:
        """Compute the generation metric
        from model outputs and reference targets.

        Args:
            outputs (list[str]): Model-generated text outputs.
            targets (list[str]):- Reference texts.
        Returns:
            float: The computed metric value.
        """

        return self._compute_raw(outputs, targets)


def create_metric(name: str) -> BaseMetric:
    """
    Create metric instance based on string name
    Supported metrics: accuracy, f1, bleu, rouge, meteor

    Args:
        name (str): Name of the metric to create
    Returns:
        BaseMetric: Instance of the specified metric
    Raises:
        ValueError: If the specified metric name is not recognized
    """

    if name in CLASSIFICATION_METRICS:
        return ClassificationMetric(name)

    if name in GENERATION_METRICS:
        return GenerationMetric(name)

    raise ValueError(f"Unknown metric: {name}")
