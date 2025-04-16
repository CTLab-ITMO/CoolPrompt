from typing import List
from src.data.base.datasets import BaseDataset
from src.evaluation.evaluator import BaseNLPEvaluator
from src.utils.eval_utils import Infer


class Cached01Scorer:

    def __init__(self, model_name, dataset_cls: BaseDataset, split, tokenizer, default_gen_args, ds_scorer: BaseNLPEvaluator, sample=100):
        self.model_name = model_name
        self.dataset_cls = dataset_cls
        
        self.split = split
        self.sample = sample

        self.tokenizer = tokenizer
        self.default_gen_args = default_gen_args
        self.ds_scorer = ds_scorer
        self.cache = {}

    def _score_prompt(self, prompt, model_gen_args):
        eval_ds = self.dataset_cls(
            tokenizer=self.tokenizer,
            split=self.split,
            prompt=prompt,
            sample=self.sample
        )

        return self.ds_scorer.evaluate_vllm_server(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            eval_ds=eval_ds,
            model_generate_args=model_gen_args
        )['f1'] # only classification for now

    def __call__(self, prompts, model_gen_args=None):

        model_gen_args = model_gen_args or {}
        model_gen_args = self.default_gen_args | model_gen_args

        prompts_to_compute = [p for p in prompts if p not in self.cache]

        computed_scores = [self._score_prompt(prompt, model_gen_args) for prompt in prompts_to_compute]
        for prompt, score in zip(prompts_to_compute, computed_scores):
            self.cache[prompt] = score

        return [self.cache[prompt] for prompt in prompts]

    def get_predictions(self, prompt: str, infer_wrapper: Infer):
        
        eval_ds : BaseDataset = self.dataset_cls(
            tokenizer=self.tokenizer,
            split=self.split,
            prompt=prompt,
            sample=self.sample
        )
        label2id = eval_ds.get_labels_mapping()
        id2label = {v: k for k, v in label2id.items()}
        token_ids = [token_id for token_id, _, _ in eval_ds]
        label_ids = [label_id for _, _, label_id in eval_ds]
        prompts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        prompts_ordered: List[str] = []
        preds_ordered: List[str] = []
        labels_ordered: List[str] = []
        
        for prompt, label_id in zip(prompts, label_ids):
            res, label_id = infer_wrapper(prompt, label_id)
            pred_id = self.ds_scorer._prepare_predictions(self.tokenizer, eval_ds, [res])[0]
            prompts_ordered.append(prompt)
            preds_ordered.append(id2label.get(pred_id, "BAD ANSWER FORMAT"))
            labels_ordered.append(id2label[label_id.item()])

        return prompts_ordered, labels_ordered, preds_ordered
