from src.evaluation.evaluator import BaseNLPEvaluator


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    pred = predictor.inference(ex, prompt)
    return prompt, ex, pred


class Cached01Scorer:

    def __init__(self, model, dataset_cls, data_path, tokenizer, ds_scorer: BaseNLPEvaluator):
        self.model = model
        self.dataset_cls = dataset_cls

        self.data_path = data_path
        self.split = split
<<<<<<< Updated upstream

=======
        
>>>>>>> Stashed changes
        self.tokenizer = tokenizer
        self.ds_scorer = ds_scorer
        self.cache = {}

    def _score_prompt(self, prompt, model_gen_args):
        eval_ds = self.dataset_cls(
            tokenizer=self.tokenizer,
            data_path=self.data_path,
            prompt=prompt
        )

        return self.ds_scorer.evaluate(
            self.model,
            tokenizer=self.tokenizer,
            eval_ds=eval_ds,
            model_generate_args=model_gen_args
        )

    def __call__(self, prompts, model_gen_args=None):

        prompts_to_compute = [p for p in prompts if p not in self.cache]

        computed_scores = [self._score_prompt(prompt, model_gen_args) for prompt in prompts_to_compute]
        for prompt, score in zip(prompts_to_compute, computed_scores):
            self.cache[prompt] = score

        return [self.cache[prompt] for prompt in prompts]