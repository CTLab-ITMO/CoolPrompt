from abc import ABC, abstractmethod
from evaluate import load
from coolprompt.utils.parsing import extract_answer
from coolprompt.utils.logging_config import logger
from coolprompt.utils.enums import Task

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

    Attributes:
        ANS_TAGS: tuple - Start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: int - Special value indicating parsing failure
    """

    ANS_TAGS = ("<ans>", "</ans>")

    def __init__(self, name: str) -> None:
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        self._name = name
        self._metric = load(name)
        self._compute_kwargs = {}

    def _compute_raw(
        self, outputs: list[str | int], targets: list[str | int]
    ) -> float:
        """Compute metric value from preprocessed model answers.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            float: Computed metric value
        """

        return self._metric.compute(
            predictions=outputs, references=targets, **self._compute_kwargs
        )[self._name]

    @abstractmethod
    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Encode labels into internal representation for both
        outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        pass

    def compute(
        self, outputs: list[str | int], targets: list[str | int]
    ) -> float:
        """Compute metric value from text model outputs

        Must be implemented by subclasses to handle input formatting.

        Args:
            outputs (list[str|int]): Model predictions (just text)
            targets (list[str|int]): Ground truth labels
        Returns:
            float: Computed metric value
        """
        output_labels = list(
            map(
                lambda x: extract_answer(
                    x, self.ANS_TAGS, self.FORMAT_MISMATCH_LABEL
                ),
                outputs,
            )
        )
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(
            output_labels, targets
        )
        return self._compute_raw(encoded_output_labels, encoded_targets)

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return self._name == other._name


class ClassificationMetric(BaseMetric):
    """Base class for classification metrics with answer parsing functionality.

    Handles extraction of labels from model outputs
    containing XML-style <ans> tags
    and label encoding for metric computation.
    """

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
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int], list[int]]:
        """Encode string labels into integer IDs for both outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        if self.label_to_id is None:
            self.extract_labels(targets)

        encoded_output_labels = [
            self.label_to_id[label] if label in self.label_to_id else -1
            for label in output_labels
        ]
        encoded_targets = [self.label_to_id[label] for label in targets]
        return encoded_output_labels, encoded_targets

    def extract_labels(self, targets: list[str | int]) -> None:
        """Extract unique labels from targets and encode them into IDs.

        Args:
            targets (list[str  |  int]): Ground truth labels.
        """

        self.label_to_id = dict()
        for x in targets:
            label = str(x)
            if label not in self.label_to_id:
                self.label_to_id[label] = len(self.label_to_id)


class GenerationMetric(BaseMetric):
    """Base class for generation metrics.

    Provides a generic implementation for metrics that compare generated text
    to reference text.
    """

    FORMAT_MISMATCH_LABEL = ""

    def __init__(self, name: str):
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        super().__init__(name)
        if name == "rouge":
            self._name = "rougeL"

    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Returns labels without encoding for generation metrics.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[str], list[str]]: input values
        """

        return output_labels, targets


def validate_and_create_metric(task: Task, metric: str | None) -> str:
    """
    Validates given metric in order to correspond the given task.
    Returns the given metric name back if the validation succeeded.

    Args:
        task (Task): The type of task, either "classification" or "generation".
        metric (str): Name of the metric to validate.
    Returns:
        str: the name of the metric.
    Raises:
        ValueError: If the specified task name is not recognized
        ValueError: If the specified metric name is not
            matched to the specified task name.
    """

    if metric is None:
        metric = get_default_metric(task)
    match task:
        case Task.CLASSIFICATION:
            if metric in CLASSIFICATION_METRICS:
                return ClassificationMetric(metric)
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {', '.join(CLASSIFICATION_METRICS)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        case Task.GENERATION:
            if metric in GENERATION_METRICS:
                return GenerationMetric(metric)
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {', '.join(GENERATION_METRICS)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    return metric


def get_default_metric(task: Task) -> str:
    """
    Returns default metric names for the provided task name.

    Args:
        task (Task): The type of task, either "classification" or "generation".
    Returns:
        str: the name of the default metric for the specified task.
    """

    match task:
        case Task.CLASSIFICATION:
            return "f1"
        case Task.GENERATION:
            return "meteor"
