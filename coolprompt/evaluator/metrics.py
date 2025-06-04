from abc import ABC, abstractmethod

from evaluate import load

TASK_TYPES = {
    "classification",
    "generation"
}

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

    def _extract_label_id_from_answer(self, answer: str) -> str | int:
        """Extract label from model output string containing XML-style tags.

        Args:
            answer (str): Model output string potentially containing <ans> tags
        Returns:
            str | int: Extracted label or FORMAT_MISMATCH_LABEL
            if parsing fails
        """

        start_tag, end_tag = self.ANS_TAGS
        start_idx = answer.rfind(start_tag)

        if start_idx == -1:
            return self.FORMAT_MISMATCH_LABEL

        content_start = start_idx + len(start_tag)
        end_idx = answer.find(end_tag, content_start)

        if end_idx == -1:
            return self.FORMAT_MISMATCH_LABEL

        label = answer[content_start:end_idx]
        return label

    def _encode_labels(self,
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

        if self.label_to_id is None:
            self.extract_labels(targets)

        encoded_output_labels = [
            self.label_to_id[label] if label in self.label_to_id
            else -1 for label in output_labels]
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

    def compute(self,
                outputs: list[str | int],
                targets: list[str | int]) -> float:
        """Compute the classification metric from model
        outputs and ground truth targets.

        This method extracts labels from outputs,
        encodes them along with targets,
        and computes the metric value.

        Args:
            outputs (list[str|int]): Model output strings.
            targets (list[str|int]): Ground truth labels.
        Returns:
            float: The computed metric value.
        """

        output_labels = list(map(self._extract_label_id_from_answer, outputs))
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


def validate_metric(task: str, metric: str | None) -> str:
    if task not in TASK_TYPES:
        raise ValueError(
            f"Invalid task type: {task}. Must be one of {TASK_TYPES}."
        )
    if metric is None:
        return get_default_metric(task)
    if task == "classification" and metric not in CLASSIFICATION_METRICS:
        raise ValueError(
            f"Invalid metric for {task} task: {metric}."
            f"Must be one of {CLASSIFICATION_METRICS}."
        )
    if task == "generation" and metric not in GENERATION_METRICS:
        raise ValueError(
            f"Invalid metric for {task} task: {metric}."
            f"Must be one of {GENERATION_METRICS}."
        )
    return metric


def get_default_metric(task: str) -> str:
    if task == "classification":
        return "f1"
    elif task == "generation":
        return "meteor"
