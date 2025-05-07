from abc import ABC, abstractmethod
from evaluate import load
import torch


class BaseMetric(ABC):
    def __init__(self, name):
        self._name = name
        self._metric = load(name)

    def _compute_raw(self, outputs, targets):
        return self._metric.compute(predictions=outputs, references=targets)[self._name]

    @abstractmethod
    def compute(self, outputs, targets):
        pass


class ClassificationMetric(BaseMetric):
    ANS_TAGS = ("<ans>", "</ans>")
    FORMAT_MISMATCH_LABEL = -1

    def _extract_label_id_from_answer(
        self, answer: str
    ):
        start_tag, end_tag = self.ANS_TAGS
        start_idx = answer.rfind(start_tag)

        if start_idx == -1:
            return torch.tensor(self.FORMAT_MISMATCH_LABEL)

        content_start = start_idx + len(start_tag)
        end_idx = answer.find(end_tag, content_start)

        if end_idx == -1:
            return self.FORMAT_MISMATCH_LABEL

        label = answer[content_start:end_idx]
        return label

    def _encode_labels(self, output_labels, targets):
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

    def compute(self, outputs, targets):
        output_labels = list(map(self._extract_label_id_from_answer, outputs))
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(output_labels, targets)
        return self._compute_raw(encoded_output_labels, encoded_targets)


class Accuracy(ClassificationMetric):
    def __init__(self):
        super().__init__("accuracy")


class F1Score(ClassificationMetric):
    def __init__(self):
        super().__init__("f1")


class GenerationMetric(BaseMetric):
    def compute(self, outputs, targets):
        return self._compute_raw(outputs, targets)


class BLEUScore(GenerationMetric):
    def __init__(self):
        super().__init__("bleu")


class ROUGEScore(GenerationMetric):
    def __init__(self):
        super().__init__("rouge")


class METEORScore(GenerationMetric):
    def __init__(self):
        super().__init__("meteor")
