from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from evaluate import load
import torch


class BaseNLPEvaluator(ABC):
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.metrics: Dict[str, any] = {}
        self._load_default_metrics()

    @abstractmethod
    def _load_default_metrics(self):
        """Load task-specific default metrics"""
        pass

    @abstractmethod
    def _compute_metrics(self, model_outputs, references) -> Dict:
        """Compute metrics for specific task"""
        pass

    @abstractmethod
    def add_batch(self,
                  model_outputs=None,
                  references=None) -> Dict:
        """
        Wrapper for evaluate.add_batch with preprocessing.
        """
        pass

    def add_metric(self, metric_name: str, **metric_args):
        """Add custom metric from HF evaluate library"""
        self.metrics[metric_name] = load(metric_name, **metric_args)

    def compute(self,
                model_outputs=None,
                references=None) -> Dict:
        """
        Template method for computing metrics
        """
        return self._compute_metrics(model_outputs, references)


class TokenClassificationEvaluator(BaseNLPEvaluator):

    CLASSIFICATION_IGNORE_IDX = -100

    def __init__(self, all_labels: List[str],
                 label_ids_to_ignore: List[int] = [CLASSIFICATION_IGNORE_IDX]):
        super().__init__(task_type="token-classification")
        self.id2label = {i: label for i, label in enumerate(all_labels)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        self.label_ids_to_ignore = label_ids_to_ignore

    def _load_default_metrics(self):
        self.metrics = {"seqeval": load("seqeval")}

    def _preprocess_inputs(self,
                           model_outputs,
                           true_label_ids) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Converts given model_outputs and label_ids
        into true_predictions and true_labels,
        skipping special tokens
        """
        if model_outputs is None or true_label_ids is None:
            return None, None

        logits = model_outputs.logits
        pred_ids = torch.argmax(logits, axis=-1).tolist()
        true_label_ids = true_label_ids.tolist()

        # Filter out ignored labels
        true_predictions = [
            [self.id2label[p]
             for p, l in zip(pred_id, label_id) if l not in self.label_ids_to_ignore]
            for pred_id, label_id in zip(pred_ids, true_label_ids)
        ]
        true_labels = [
            [self.id2label[id]
             for id in label_id if id not in self.label_ids_to_ignore]
            for label_id in true_label_ids
        ]

        return true_predictions, true_labels

    def _compute_metrics(self, model_outputs, true_label_ids):
        true_predictions, true_labels = (
            self._preprocess_inputs(model_outputs, true_label_ids))

        return self.metrics["seqeval"].compute(
            predictions=true_predictions,
            references=true_labels
        )

    def add_batch(self,
                  model_outputs=None,
                  references=None) -> Dict:
        """
        Wrapper for evaluate.add_batch with preprocessing.
        """
        true_predictions, true_labels = self._preprocess_inputs(
            model_outputs, references)

        return self.metrics["seqeval"].add_batch(
            predictions=true_predictions,
            references=true_labels
        )
