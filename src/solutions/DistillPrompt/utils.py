"""
Caching Evaluator Module

This module provides the CachingEvaluator class, which is used to evaluate prompts
with caching to avoid redundant computations. It also includes a utility function
to set random seeds for reproducibility.
"""


import os
import random
import typing as tp

import numpy as np
import torch
from vllm import LLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.evaluation.evaluator import BaseNLPEvaluator
from src.utils.eval_utils import create_ds_from_task


class CachingEvaluator:
    """
    A class for evaluating prompts with caching to improve efficiency.

    This class evaluates prompts using a language model and caches the results
    to avoid redundant computations for the same prompt and evaluation settings.
    """

    def __init__(self, task: str, model: LLM, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 base_evaluator: BaseNLPEvaluator, default_gen_args: dict[str, tp.Any],  batch_size: int = 100):
        """
        Initializes the CachingEvaluator with the necessary components.

        Args:
            task (str): The name of the task for which prompts are evaluated.
            model (LLM): The language model used for evaluation.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer for processing text.
            base_evaluator (BaseNLPEvaluator): The evaluator used to assess the model's performance.
            default_gen_args (dict[str, tp.Any]): Default arguments for model generation.
            batch_size (int): The batch size for evaluation.
        """
        self.task_name = task
        self.model = model
        self.tokenizer = tokenizer

        self.base_evaluator = base_evaluator
        self.default_gen_args = default_gen_args
        self.batch_size = batch_size


        self.cache = {}

    def _score_prompt(self, prompt, model_gen_args, split, sample):
        
        eval_ds = create_ds_from_task(
            self.task_name,
            tokenizer=self.tokenizer,
            split=split,
            prompt=prompt,
            sample=sample
        )
        
        metrics = self.base_evaluator.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=eval_ds,
            batch_size=self.batch_size,
            model_generate_args=model_gen_args
        )
        
        return metrics

    def __call__(self, prompt: str, split='train', sample=100) -> dict[str, float]:
        """
        Evaluates a prompt and returns cached results if available.

        Args:
            prompt (str): The prompt to be evaluated.
            split (str): The dataset split to use for evaluation ('train' or 'test').
            sample (int): The number of samples to use for evaluation.

        Returns:
            dict[str, float]: A dictionary of evaluation metrics.
        """
        if (prompt, split, sample) not in self.cache:
            self.cache[(prompt, split, sample)] = self._score_prompt(prompt, self.default_gen_args, split, sample)
        
        return self.cache[(prompt, split, sample)]


def seed_everyting(seed: int = 42):
    """
    Sets random seeds for various libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
