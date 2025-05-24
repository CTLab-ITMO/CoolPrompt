from abc import ABC, abstractmethod
from typing import List, Tuple
from evaluate import load

type InputType = List[str] | List[int]
type LabelType = str | int


class BaseMetric(ABC):
    """Abstract base class for implementing evaluation metrics.
    
    Provides common infrastructure for loading metrics from HuggingFace's evaluate
    library and defining metric computation interfaces.    
    """
    
    def __init__(self, name: str) -> None:
        """Initialize metric with specified evaluate library metric name.
        
        Args:
            name: str - Name of metric to load from evaluate library
        """
        
        self._name = name
        self._metric = load(name)

    def _compute_raw(self, outputs: InputType, targets: InputType) -> float:
        """Compute metric value from preprocessed model answers.
        
        Args:
            outputs: InputType - Model predictions (text for generation, labels for classification)
            targets: InputType - Ground truth labels
            
        Returns:
            float - Computed metric value
        """
        
        return self._metric.compute(predictions=outputs, references=targets)[self._name]

    @abstractmethod
    def compute(self, outputs: InputType, targets: InputType) -> float:
        """Compute metric value from text model outputs
        
        Must be implemented by subclasses to handle input formatting.

        Args:
            outputs: InputType - Model predictions (just text)
            targets: InputType - Ground truth labels
            
        Returns:
            float - Computed metric value
        """
        pass


class ClassificationMetric(BaseMetric):
    """Base class for classification metrics with answer parsing functionality.
    
    Handles extraction of labels from model outputs containing XML-style <ans> tags
    and label encoding for metric computation.
    
    Attributes:
        ANS_TAGS: tuple - Start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: int - Special value indicating parsing failure
    """
    
    ANS_TAGS = ("<ans>", "</ans>")
    FORMAT_MISMATCH_LABEL = -1

    def _extract_label_id_from_answer(self, answer: str) -> LabelType:
        """Extract label from model output string containing XML-style tags.
        
        Args:
            answer: str - Model output string potentially containing <ans> tags
            
        Returns:
            LabelType - Extracted label or FORMAT_MISMATCH_LABEL if parsing fails
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

    def _encode_labels(self, output_labels: InputType, targets: InputType) -> Tuple[List[int], List[int]]:
        """Encode string labels into integer IDs for both outputs and targets.

        Args:
            output_labels: InputType - Extracted labels from model outputs.
            targets: InputType - Ground truth labels.

        Returns:
            Tuple[List[int], List[int]]: Encoded output labels and encoded targets.
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

    def compute(self, outputs: InputType, targets: InputType) -> float:
        """Compute the classification metric from model outputs and ground truth targets.

        This method extracts labels from outputs, encodes them along with targets,
        and computes the metric value.

        Args:
            outputs: InputType - Model output strings.
            targets: InputType - Ground truth labels.

        Returns:
            float: The computed metric value.
        """
        
        output_labels = list(map(self._extract_label_id_from_answer, outputs))
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(output_labels, targets)
        return self._compute_raw(encoded_output_labels, encoded_targets)


class Accuracy(ClassificationMetric):
    """Accuracy metric for classification tasks"""
    
    def __init__(self) -> None:
        super().__init__("accuracy")


class F1Score(ClassificationMetric):
    """F1 score metric for classification tasks"""
    
    def __init__(self) -> None:
        super().__init__("f1")


class GenerationMetric(BaseMetric):
    """Base class for generation metrics.

    Provides a generic implementation for metrics that compare generated text
    to reference text.
    """
    
    def compute(self, outputs: InputType, targets: InputType) -> float:
        """Compute the generation metric from model outputs and reference targets.

        Args:
            outputs: List[str] - Model-generated text outputs.
            targets: List[str] - Reference texts.

        Returns:
            float: The computed metric value.
        """
        
        return self._compute_raw(outputs, targets)


class BLEUScore(GenerationMetric):
    """BLEU score metric for text generation tasks"""
    
    def __init__(self) -> None:
        super().__init__("bleu")


class ROUGEScore(GenerationMetric):
    """ROUGE score metric for text generation tasks"""
    
    def __init__(self) -> None:
        super().__init__("rouge")


class METEORScore(GenerationMetric):
    """METEOR score metric for text generation tasks"""
    
    def __init__(self) -> None:
        super().__init__("meteor")
