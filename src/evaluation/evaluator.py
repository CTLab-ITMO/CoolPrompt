"""
NLP Evaluation Framework

Provides base classes and concrete implementations for evaluating NLP models
on different tasks with standardized metrics and output formatting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from evaluate import load, EvaluationModule
from tqdm import tqdm
import concurrent.futures
import torch
from vllm import LLM, SamplingParams

from src.data.base.datasets import BaseDataset
from src.utils.eval_utils import Infer


class BaseNLPEvaluator(ABC):
    """Abstract base class for NLP model evaluation.

    Implements the common evaluation workflow while allowing task-specific
    implementations through abstract methods.

    Attributes:
        task_type: String identifier for the task type (e.g., 'text-classification')
        metrics: Dictionary of loaded evaluation metrics
    """

    def __init__(self):
        """Initialize base evaluator.
        
        """
        self.metrics: Dict[str, EvaluationModule] = {}
        self._load_default_metrics()

    @abstractmethod
    def _load_default_metrics(self):
        """Load task-specific default metrics

        Should populate self.metrics with metric names as keys and
        loaded EvaluationModule instances as values.
        """
        pass

    @abstractmethod
    def _compute_metrics(self, compute_metric_list: list[str] = []) -> Dict[str, float | None]:
        """Compute and return all configured metrics.
        
        Args:
            compute_metric_list: Optional list of metrics to compute,
            used during training to reduce computations.

        Returns:
            Dictionary mapping metric names to computed values
        """
        pass

    def _add_batch(self, predictions=None, references=None) -> None:
        """
        Wrapper for evaluate.add_batch for a list of metrics.
        """
        for metric in self.metrics.values():
            metric.add_batch(predictions=predictions, references=references)

    @abstractmethod
    def _prepare_labels(
        self, tokenizer: Any, dataset: BaseDataset, label_ids: torch.Tensor
    ) -> List[Any]:
        """Process batch of ground truth labels into final format.

        Args:
            tokenizer: Model tokenizer used for decoding
            dataset: Dataset used for evaluation
            label_ids: Tensor containing label IDs from dataset

        Returns:
            List of processed labels in metric-compatible format
        """
        pass

    @abstractmethod
    def _get_max_tokens(self) -> int:
        pass

    def _get_default_generation_args_vllm(self, tokenizer) -> Dict[str, Any]:
        """Default arguments for vllm model generation"""
        return {
            "temperature": 0.0,
            "max_tokens": self._get_max_tokens(),
            "stop_token_ids": [tokenizer.eos_token_id],
        }

    def _get_default_generation_args_hf(self, tokenizer) -> Dict[str, Any]:
        """Default arguments for hugginface model generation"""
        return {
            "temperature": 0.0,
            "max_new_tokens": self._get_max_tokens(),
            "eos_token_id": [tokenizer.eos_token_id],
        }

    @abstractmethod
    def _prepare_predictions(
        self, tokenizer: Any, dataset: BaseDataset, outputs: List[str]
    ) -> List[Any]:
        """Convert generated tokens into formatted predictions.

        Args:
            tokenizer: Model tokenizer used for decoding
            dataset: Dataset used for evaluation
            outputs: List of strings containing detokenized model outputs

        Returns:
            List of processed predictions in metric-compatible format
        """
        pass

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        eval_ds: BaseDataset,
        compute_metric_list: list[str] = [],
        batch_size: int = 64,
        model_generate_args: Dict[str, Any] = None,
    ) -> Dict[str, float | None]:
        """Execute full evaluation workflow with HF model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer matching the model
            eval_ds: Evaluation dataset
            compute_metric_list: Metrics to compute, if empty all class metrics are computed
            batch_size: Batch size for evaluation
            model_generate_args: Additional arguments for model generation

        Returns:
            Dictionary of computed metrics
        """

        generate_args = self._get_default_generation_args_hf(tokenizer)

        if model_generate_args:
            generate_args.update(model_generate_args)

        val_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)

        for input_ids, attention_mask, label_ids in tqdm(val_dataloader):
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generate_args
            )

            generated_tokens = [
                output[len(ids) :] for output, ids in zip(outputs, input_ids)
            ]

            generated_strings = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            predictions = self._prepare_predictions(
                tokenizer, eval_ds, generated_strings
            )

            labels = self._prepare_labels(tokenizer, eval_ds, label_ids)

            self._add_batch(predictions, labels)

        return self._compute_metrics(compute_metric_list)

    def evaluate_vllm(
        self,
        model: LLM,
        tokenizer: Any,
        eval_ds: BaseDataset,
        compute_metric_list: list[str] = [],
        batch_size: int = 64,
        model_generate_args: Dict[str, Any] = None,
    ) -> Dict[str, float | None]:
        """Execute full evaluation workflow with vllm.LLM model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer matching the model
            eval_ds: Evaluation dataset
            compute_metric_list: Metrics to compute, if empty all class metrics are computed
            batch_size: Batch size for evaluation
            model_generate_args: Additional arguments for model generation

        Returns:
            Dictionary of computed metrics
        """

        generate_args = self._get_default_generation_args_vllm(tokenizer)

        if model_generate_args:
            generate_args.update(model_generate_args)

        sampling_params = SamplingParams(**generate_args)

        val_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)

        for input_ids, attention_mask, label_ids in tqdm(val_dataloader):

            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            answers = model.generate(
                prompts=inputs, sampling_params=sampling_params, use_tqdm=False
            )

            outputs = [answer.outputs[0].text for answer in answers]

            predictions = self._prepare_predictions(tokenizer, eval_ds, outputs)

            labels = self._prepare_labels(tokenizer, eval_ds, label_ids)

            self._add_batch(predictions, labels)

        return self._compute_metrics(compute_metric_list)

    def evaluate_vllm_server(
        self,
        model_name: str,
        tokenizer: Any,
        eval_ds: BaseDataset,
        compute_metric_list: list[str] = [],
        server_url: str = "http://localhost:8000/v1/completions",
        batch_size: int = 64,
        max_workers=16,
        model_generate_args: Dict[str, Any] = {},
    ) -> Dict[str, float | None]:
        """Execute full evaluation workflow with vllm server.

        Args:
            model_name: Name of the model to evaluate
            tokenizer: Tokenizer matching the model
            eval_ds: Evaluation dataset
            compute_metric_list: Metrics to compute, if empty all class metrics are computed
            server_url: Vllm server url
            batch_size: Batch size for evaluation
            max_workers: Number of workers for async requests
            model_generate_args: Additional arguments for model generation

        Returns:
            Dictionary of computed metrics
        """

        generate_args = self._get_default_generation_args_vllm(tokenizer)

        if model_generate_args:
            generate_args.update(model_generate_args)

        val_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)

        infer_fn = Infer(model_name, server_url, model_generate_args)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for input_ids, attention_mask, label_ids in tqdm(val_dataloader):

                inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                answers = []
                ordered_label_ids = []

                futures = [
                    executor.submit(infer_fn, prompt, label_id.tolist())
                    for prompt, label_id in zip(inputs, label_ids)
                ]

                for future in concurrent.futures.as_completed(futures):
                    answer, label_id = future.result()
                    answers.append(answer)
                    ordered_label_ids.append(label_id)

                ordered_label_ids = torch.tensor(ordered_label_ids)

                predictions = self._prepare_predictions(tokenizer, eval_ds, answers)

                labels = self._prepare_labels(tokenizer, eval_ds, ordered_label_ids)

                self._add_batch(predictions, labels)

        return self._compute_metrics(compute_metric_list)


class TextClassificationEvaluator(BaseNLPEvaluator):
    """Evaluator for text classification tasks.

    Handles label extraction from formatted model outputs and computes
    classification metrics (F1-score by default).

    Attributes:
        ANS_TAGS: Tuple containing start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: Special value indicating formatting errors
    """

    ANS_TAGS: Tuple[str, str] = ("<ans>", "</ans>")
    FORMAT_MISMATCH_LABEL: int = -1

    def _load_default_metrics(self):
        self.metrics = {
            "f1": load("f1"),
            "accuracy": load("accuracy"),
        }

    def _compute_metrics(self, compute_metric_list: list[str] = []) -> Dict[str, Optional[float]]:
        if not compute_metric_list:
            compute_metric_list = ["f1", "accuracy"]
            
        results = {
            "f1": None,
            "accuracy": None
        }
        
        if "f1" in compute_metric_list:
            results["f1"] = self.metrics["f1"].compute(average="macro")["f1"]
        
        if "accuracy" in compute_metric_list:
            results["accuracy"] = self.metrics["accuracy"].compute()["accuracy"]
        
        return results

    def _get_max_tokens(self) -> int:
        return 50

    def _prepare_labels(self, tokenizer, eval_ds, label_ids) -> Any:
        return label_ids

    def _prepare_predictions(self, tokenizer, eval_ds, generated_strings) -> Any:
        label2id = eval_ds.get_labels_mapping()

        predictions = [
            self._extract_label_id_from_answer(answer, label2id)
            for answer in generated_strings
        ]

        return predictions

    def _extract_label_id_from_answer(
        self, answer: str, label2id: dict[str, int]
    ) -> torch.Tensor:
        """Parse label from formatted answer string.

        Args:
            answer: Model-generated answer string (only newly generated tokens are included)
            label2id: Dataset label2id mapping.

        Returns:
            Numerical label ID or FORMAT_MISMATCH_LABEL (-1) on errors
        """
        start_tag, end_tag = self.ANS_TAGS
        start_idx = answer.rfind(start_tag)

        if start_idx == -1:
            return torch.tensor(self.FORMAT_MISMATCH_LABEL)

        content_start = start_idx + len(start_tag)
        end_idx = answer.find(end_tag, content_start)

        if end_idx == -1:
            return torch.tensor(self.FORMAT_MISMATCH_LABEL)

        label = answer[content_start:end_idx]

        label_id = label2id.get(label, self.FORMAT_MISMATCH_LABEL)

        return torch.tensor(label_id)


class GenerationEvaluator(BaseNLPEvaluator):

    def _load_default_metrics(self):

        self.metrics = { 
            "bleu": load("bleu"),
            "rouge": load("rouge"),  
            "meteor": load("meteor"),  
        }

    def _compute_metrics(self, compute_metric_list: list[str]=[]) -> Dict[str, Optional[float]]:
        """Compute all configured generation metrics.

        Returns:
            Dictionary containing:
            - rouge: ROUGE-L score
            - bleu: BLEU score
            - meteor: METEOR score
        """
        
        if not compute_metric_list:
            compute_metric_list = ["bleu", "rouge", "meteor"]
            
        results = {
            "bleu": None,
            "rouge": None,
            "meteor": None
        }
        
        if "bleu" in compute_metric_list:
            results["bleu"] = self.metrics["bleu"].compute()["bleu"]
        
        if "rouge" in compute_metric_list:
            results["rouge"] = self.metrics["rouge"].compute()["rougeL"]
            
        if "meteor" in compute_metric_list:
            results["meteor"] = self.metrics["meteor"].compute()["meteor"]
        
        return results
        

    def _get_max_tokens(self) -> int:
        return 128

    def _prepare_labels(self, tokenizer, eval_ds, label_ids) -> Any:
        return tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    def _prepare_predictions(self, tokenizer, eval_ds, generated_strings) -> Any:
        return generated_strings
