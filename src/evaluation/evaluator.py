from abc import ABC, abstractmethod
from typing import Dict, Any

from src.data.base.datasets import BaseClassificationDataset, BaseDataset
from evaluate import load
from tqdm import tqdm
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

    @abstractmethod
    def evaluate(self, model: Any, tokenizer: Any, eval_ds: BaseDataset, batch_size: int = 64):
        """
        Template method for computing metrics
        """
        pass


class TextClassificationEvaluator(BaseNLPEvaluator):
    ANS_BEGINNING_TAG = "<ans>"
    ANS_ENDING_TAG = "</ans>"

    FORMAT_MISMATCH_LABEL = -1

    def __init__(self):
        super().__init__(task_type="token-classification")

    def _load_default_metrics(self):
        self.metrics = {"f1": load("f1")}


    def _compute_metrics(self,
                  predictions=None,
                  references=None) -> float:

        return self.metrics["f1"].compute(
            predictions=predictions,
            references=references,
            average='macro'
        )["f1"]

    def add_batch(self,
                  predictions=None,
                  references=None) -> Dict:
        """
        Wrapper for evaluate.add_batch with preprocessing.
        """

        return self.metrics["f1"].add_batch(
            predictions=predictions,
            references=references
        )

    def _extract_label_id_from_answer(self, answer: str, label2id: dict[str, int]) -> torch.Tensor:

        start_idx = answer.rfind(self.ANS_BEGINNING_TAG)

        if start_idx == -1:
            return torch.tensor(self.FORMAT_MISMATCH_LABEL)

        content_start = start_idx + len(self.ANS_BEGINNING_TAG)
        end_idx = answer.find(self.ANS_ENDING_TAG, content_start)

        if end_idx == -1:
            return torch.tensor(self.FORMAT_MISMATCH_LABEL)

        label = answer[content_start:end_idx]

        label_id = label2id.get(label, self.FORMAT_MISMATCH_LABEL)

        return torch.tensor(label_id)

    def evaluate(self, model: Any, tokenizer: Any, eval_ds: BaseClassificationDataset, batch_size: int = 64,
                 model_generate_args=None):
        """
        Template method for computing metrics
        """
        if model_generate_args is None:
            model_generate_args = {}

        val_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)

        label2id = eval_ds.get_labels_mapping()

        if not model_generate_args:
            model_generate_args = {
                "eos_token_id": tokenizer.eos_token_id
            }

        bad_format = 0
        total = 0

        for input_ids, attention_mask, labels in tqdm(val_dataloader):
            # generating batch output
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_generate_args
            )

            pure_responses = [
                output[len(ids):] for output, ids in zip(outputs, input_ids)
            ]

            # decoding answers
            answers = tokenizer.batch_decode(pure_responses, skip_special_tokens=True)

            # parsing answers
            predictions = [self._extract_label_id_from_answer(answer, label2id)
                           for answer in answers]

            bad_format_preds = [id.item() for id in predictions if id.item() == -1]

            bad_format += len(bad_format_preds)
            total += len(predictions)


            self.add_batch(predictions, labels)

        print("Wront output format %:", (bad_format / total) * 100)

        f1_macro = self._compute_metrics()
        return f1_macro



class GenerationEvaluator(BaseNLPEvaluator):

    def __init__(self):
        super().__init__(task_type="text-generation")

    def _load_default_metrics(self):
        self.metrics = {
            "bleu": load("bleu"),
            "rouge": load("rouge"),
            "meteor": load("meteor")
        }

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "bleu": self.metrics["bleu"].compute(predictions=self.predictions, references=self.references)["bleu"],
            "rouge": self.metrics["rouge"].compute(predictions=self.predictions, references=self.references)["rougeL"],
            "meteor": self.metrics["meteor"].compute(predictions=self.predictions, references=self.references)["meteor"]
        }

    def add_batch(self, predictions=None, references=None):
        for metric in self.metrics.values():
            metric.add_batch(predictions, references)


    def evaluate(self, model, tokenizer, eval_ds, batch_size=64, model_generate_args=None):
        if model_generate_args is None:
            model_generate_args = {
                "max_new_tokens": 128,
                "eos_token_id": tokenizer.eos_token_id
            }

        dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)
        bad_format_count = 0
        total = 0

        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                **model_generate_args
            )

            # Remove input context from generated text
            generated = [output[len(input):] for input, output in zip(inputs, outputs)]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Extract formatted responses
            processed = []
            for text in decoded:
                clean = self._extract_generated_text(text)
                if clean is None:
                    bad_format_count += 1
                    clean = ""  # Empty string for failed formats
                processed.append(clean)
                total += 1

            self.add_batch(predictions=processed, references=batch["target_text"])

        print(f"Bad format ratio: {bad_format_count / total * 100:.2f}%")
        return self._compute_metrics()


