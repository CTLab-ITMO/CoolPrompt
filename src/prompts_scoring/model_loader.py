import contextlib
import gc
import json
import logging
import os
import random
from typing import Any

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.evaluation.evaluator import (
    GenerationEvaluator,
    TextClassificationEvaluator,
)
from src.utils.data import (
    INNER_GENERATION_TASKS,
)
from src.utils.load_dataset import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Model loader class using for the given model's initialization
    and prompts evaluation. Initializes model via vllm from hf repo's name.
    """

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] = None,
        batch_size: int = 16,
        verbose: int = 1,
    ):
        """Initializes the model without given task.

        Args:
            model_name: hf repo's name for the model
            model_config (optional): explicit config for vllm initialization
            (such as gpu_memory_utilization)
            batch_size (optional): using batch size. Defaults to 16
            verbose (optional): specializes the logging level: 0 for errors,
            1 for info, 2 for debug.

        Raises:
            ValueError: occurs when unable to collect and open json files with
            basic prompts and task labels.
        """
        self.model_name = model_name
        self.task_name = None
        self.bench_name = None
        self._batch_size = None
        self._model = None
        self._tokenizer = None
        self._terminators = None
        self._device = None
        self._evaluator = None
        self._max_new_tokens = None
        self._labels = None
        self._base_prompt = None
        self._batch_size = batch_size

        match verbose:
            case 0:
                logger.setLevel(logging.ERROR)
            case 1:
                logger.setLevel(logging.INFO)
            case 2:
                logger.setLevel(logging.DEBUG)
        logger.info(f"Starting model {self.model_name} initialization...")

        try:
            with open("../../data/labels.json", "r") as f:
                self.labels_json = json.load(f)
            with open("../../data/basic_prompts.json", "r") as f:
                self.prompts_json = json.load(f)
        except Exception as e:
            logger.error(f"Error while opening data json files: {str(e)}")
            raise ValueError

        self.seed_everything(42)
        torch.cuda.empty_cache()
        gc.collect()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self._terminators = [self.tokenizer.eos_token_id]
        config = {
            "model": self.model_name,
            "dtype": torch.float16,
            "trust_remote_code": True,
        }
        if model_config is not None:
            config.update(model_config)
        self._model = LLM(**config)

        logger.info("Model loaded via vllm")
        logger.info("Model initializing completed")
        self.print_gpu_memory()

    def initialize(self, task: str):
        """Initializes evaluator, task labels and basic prompt
        based on the given task name.

        Args:
            task: task name, maybe with benchmark's name
            (ex. sst2, bbh/word_sorting, natural_instructions/task021)
        """

        if "/" in task:
            bench_name, task_name = task.split("/")
        else:
            bench_name, task_name = None, task

        self.task_name = task_name
        self.bench_name = bench_name

        if (
            self.task_name in ["gsm8k", "math", "samsum"]
            or self.task_name in INNER_GENERATION_TASKS
        ):
            self._evaluator = GenerationEvaluator()
            self._max_new_tokens = 250
        else:
            self._evaluator = TextClassificationEvaluator()
            self._max_new_tokens = 50
        logger.info(f"Evaluator loaded: {type(self._evaluator).__name__}")

        labels_json = self.labels_json
        try:
            if self.bench_name is not None:
                self._labels = labels_json[self.bench_name][self.task_name]
            else:
                self._labels = labels_json[self.task_name]
        except KeyError:
            self._labels = []

        prompts_json = self.prompts_json
        if self.bench_name is not None:
            self._base_prompt = prompts_json[self.bench_name][self.task_name]
        else:
            self._base_prompt = prompts_json[self.task_name]
        logger.info(f"Task {self.task_name} initialized")

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_generate_args_hf(self):
        return {
            "max_new_tokens": self._max_new_tokens,
            "eos_token_id": self._terminators,
        }

    @property
    def model_generate_args(self):
        return {
            "max_tokens": self._max_new_tokens,
            "stop_token_ids": self._terminators,
            "temperature": 0,
        }

    @property
    def device(self):
        return self._device

    def load_data(
        self,
        prompt: str,
        split: str = "train",
        full: bool = False,
        sample: int = 100,
    ):
        """Loads dataset with the provided prompt and specified options

        Args:
            prompt: prompt candidate
            split (optional): dataset splitting mode
            (either 'test' or 'train'). Defaults to 'train'
            full (optional): specifies if loading full dataset
            or only a sample. Defaults to `False`
            sample (optional): specifies the amount of dataset's
            instances. Defaults to 100"""

        logger.debug(f"Loading data for {self.task_name} dataset")
        return load_dataset(
            self.task_name,
            tokenizer=self._tokenizer,
            sample=sample if not full else None,
            split=split,
            prompt=prompt,
            device=self._device,
        )

    @property
    def labels(self):
        return self._labels

    @property
    def base_prompt(self):
        return self._base_prompt

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def batch_size(self):
        return self._batch_size

    def seed_everything(self, seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def print_gpu_memory(self):
        if not torch.cuda.is_available():
            logger.debug("CUDA not available")
            return

        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            logger.debug(
                (
                    f"GPU {i}: "
                    f"Used = {used / 1024 ** 3:.2f} GB | "
                    "Free = {free/1024**3:.2f} GB | "
                    "Total = {total/1024**3:.2f} GB"
                )
            )

    def get_metrics(
        self, candidate: str, split: str = "train", full: bool = False
    ):
        """Returns evaluation metrics for the given candidate prompt:
        accuracy/f1 for classification tasks, bleu/rouge/meteor for generation.

        Args:
            candidate: candidate prompt for loading dataset with
            split (optional): dataset splitting mode. Defaults to 'train'
            full (optional): specifies if using the full dataset
            or only a sample. Defaults to `False`
        """
        return self.evaluator.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=self.load_data(candidate, split, full),
            batch_size=self.batch_size,
            model_generate_args=self.model_generate_args,
        )

    def generate(self, prompts: str | list[str]):
        """Generates from given prompt or list of prompts
        with the loader's sampling params.

        Args:
            prompts: prompt or list of prompts
        """

        if isinstance(prompts, str):
            prompts = [prompts]
        return self.model.generate(
            prompts, sampling_params=SamplingParams(**self.model_generate_args)
        )

    def get_sample(self):
        """Gets sample from initialized task's dataset."""

        sample = self.load_data(sample=1, prompt=None)
        input_ids, _, label_id = sample[0]
        input = self._tokenizer.decode(input_ids, skip_special_tokens=True)
        question_start = input[
            input.rfind("The input:\n") + len("The input:\n") :
        ]
        question = question_start[: question_start.find("\n")]
        label2id = sample.get_labels_mapping()
        if len(label2id) != 0:
            answer = {v: k for k, v in label2id.items()}[label_id.item()]
        else:
            answer = self._tokenizer.decode(label_id, skip_special_tokens=True)
        return {"question": question, "answer": answer}

    def destroy(self):
        """Destroys the initialized model and corresponding components."""

        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        if hasattr(self._model.llm_engine, "model_executor"):
            del self._model.llm_engine.model_executor
        del self._model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
