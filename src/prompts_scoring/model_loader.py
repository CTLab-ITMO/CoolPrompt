import gc
import logging
import os
import random
from typing import Iterable

import numpy as np
import torch

from coolprompt.evaluator.evaluator import Evaluator
from coolprompt.evaluator.metrics import validate_and_create_metric
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.var_validation import validate_task

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Model loader class using for model's initialization
    and prompts evaluation. Initializes the DefaultLLM model.
    """

    def __init__(
        self,
        batch_size: int = 16,
        verbose: int = 1,
    ) -> None:
        """Initializes the model without given task.

        Args:
            batch_size (int, optional): using batch size. Defaults to 16
            verbose (int, optional): specializes the logging level:
                0 for errors, 1 for info, 2 for debug.

        Raises:
            ValueError: if JSON files with basic prompts and task labels
                cannot be read.
        """
        self._task = None
        self._metric = None
        self._batch_size = None
        self._model = None
        self._evaluator = None
        self._batch_size = batch_size

        match verbose:
            case 0:
                logger.setLevel(logging.ERROR)
            case 1:
                logger.setLevel(logging.INFO)
            case 2:
                logger.setLevel(logging.DEBUG)
        logger.info(f"Starting model {self.model_name} initialization...")

        self.seed_everything(42)
        torch.cuda.empty_cache()
        gc.collect()

        self._model = DefaultLLM.init()

        logger.info("Model initializing completed")
        self.print_gpu_memory()

    def initialize(self, task: str, metric: str) -> None:
        """Initializes evaluator with given task and metric.

        Args:
            task (str): task type (either generation or classification).
            metric (str): metric name (f1/accuracy for classification,
                bleu/rouge/meteor for generation).
        """
        self._task = validate_task(task)
        self._metric = validate_and_create_metric(task, metric)
        self._evaluator = Evaluator(self._model, task, metric)

    @property
    def model(self):
        return self._model

    @property
    def evaluator(self):
        return self._evaluator

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
        self,
        candidate: str,
        dataset: Iterable,
        target: Iterable,
        full: bool = False,
    ) -> float:
        """Returns evaluation metrics for given candidate prompt,
        dataset and target.

        Args:
            candidate (str): candidate prompt used in dataset's loading
            dataset (Iterable): evaluation dataset
            target (Iterable): target objects for evaluation
            full (bool, optional): specifies if using the whole dataset or
                only a sample of 100 instances. Defaults to `False`
        """

        template = (
            CLASSIFICATION_TASK_TEMPLATE_HYPE
            if self._task == Task.CLASSIFICATION
            else GENERATION_TASK_TEMPLATE_HYPE
        )

        if full:
            dataset_sample, target_sample = dataset, target
        else:
            total_size = len(dataset)
            n = min(100, total_size)
            indices = random.sample(range(total_size), n)
            dataset_list, target_list = list(dataset), list(target)
            dataset_sample = [dataset_list[i] for i in indices]
            target_sample = [target_list[i] for i in indices]

        return self._evaluator.evaluate(
            prompt=candidate,
            dataset=dataset_sample,
            target=target_sample,
            template=template,
        )
