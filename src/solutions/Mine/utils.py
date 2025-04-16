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

    def __init__(self, task: str, model: LLM, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 base_evaluator: BaseNLPEvaluator, default_gen_args: dict[str, tp.Any],  batch_size: int = 100):
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
        
        if (prompt, split) not in self.cache:
            self.cache[(prompt, split)] = self._score_prompt(prompt, self.default_gen_args, split, sample)
        
        return self.cache[(prompt, split)]


def seed_everyting(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
