from typing import List

import torch
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import BaseNLPEvaluator

from src.utils.eval_utils import create_ds_from_task


class Cached01Scorer:

    def __init__(self, model: LLM, task: str, split, tokenizer, default_gen_args, ds_scorer: BaseNLPEvaluator, batch_size=16, sample=100):
        self.model = model
        self.task_name = task
        
        self.split = split
        self.sample = sample

        self.tokenizer = tokenizer
        self.default_gen_args = default_gen_args
        self.ds_scorer = ds_scorer
        self.batch_size = batch_size
        self.cache = {}

    def _score_prompt(self, prompt, model_gen_args):
        
        eval_ds = create_ds_from_task(
            self.task_name,
            tokenizer=self.tokenizer,
            split=self.split,
            prompt=prompt,
            sample=self.sample
        )
        
        metrics = self.ds_scorer.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=eval_ds,
            batch_size=self.batch_size,
            model_generate_args=model_gen_args
        )
        
        if self.split == 'train':
            return metrics['f1'] # only classification for now
        
        return metrics

    def __call__(self, prompts, model_gen_args=None):

        model_gen_args = model_gen_args or {}
        model_gen_args = self.default_gen_args | model_gen_args

        prompts_to_compute = [p for p in prompts if p not in self.cache]

        computed_scores = [self._score_prompt(prompt, model_gen_args) for prompt in prompts_to_compute]
        for prompt, score in zip(prompts_to_compute, computed_scores):
            self.cache[prompt] = score

        return [self.cache[prompt] for prompt in prompts]

    def get_predictions(self, prompt: str):
        
        eval_ds = create_ds_from_task(
            self.task_name,
            tokenizer=self.tokenizer,
            split=self.split,
            prompt=prompt,
            sample=self.sample
        )
        
        label2id = eval_ds.get_labels_mapping()
        id2label = {v: k for k, v in label2id.items()}
        prompts_ordered: List[str] = []
        preds_ordered: List[str] = []
        labels_ordered: List[str] = []
        
        val_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=self.batch_size)
        
        sampling_params = SamplingParams(**self.default_gen_args)

        
        for input_ids, attention_mask, label_ids in val_dataloader:

            inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            answers = self.model.generate(
                prompts=inputs, sampling_params=sampling_params, use_tqdm=False
            )

            outputs = [answer.outputs[0].text for answer in answers]
            
            preds = self.ds_scorer._prepare_predictions(self.tokenizer, eval_ds, outputs)
            
            prompts_ordered.extend(inputs)

            pred_labels = [id2label.get(pred_id.item(), "BAD ANSWER FORMAT") for pred_id in preds]
            
            preds_ordered.extend(pred_labels)

            print("Bad answer %:", 100 * len([a for a in pred_labels if a == "BAD ANSWER FORMAT"]) / len(pred_labels) )
            
            labels_ordered.extend([
                id2label[label_id.item()] for label_id in label_ids
            ])

        return prompts_ordered, labels_ordered, preds_ordered
