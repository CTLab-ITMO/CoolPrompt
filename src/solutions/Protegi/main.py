import os

from transformers import AutoTokenizer

import evaluators
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers

from src.data.classification import SST2Dataset
from src.evaluation.evaluator import TextClassificationEvaluator


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')


TASK_TO_DS_MAP = {
    "sst-2": SST2Dataset,
}

TASK_TO_SCORER_MAP = {
    "sst-2": TextClassificationEvaluator,
}

def get_task_ds(data_dir, task_name, split, tokenizer):
    data_path = f"{data_dir}/{task_name}/{split}-00000-of-00001.parquet"

    return TASK_TO_DS_MAP[task_name](
        tokenizer=tokenizer,
        data_path=data_path
    )




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sst-2')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']

    tokenizer = AutoTokenizer.from_pretrained(args.engine)

    model = AutoModelForCausalLM.from_pretrained(args.engine)

    train_scorer = scorers.Cached01Scorer(
        model,
        dataset_cls=TASK_TO_DS_MAP[args.task],
        data_path=args.data_dir,
        tokenizer=tokenizer,
        ds_scorer=TASK_TO_SCORER_MAP[args.task]
    )

    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)


    if os.path.exists(args.out):\
        
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [train_ds.prompt]

    epochs = config['rounds'] + 1

    for round in tqdm(range(epochs)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

        # score candidates
        scores = optimizer.score_candidates(candidates, gpt4, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'{time.time() - start}\n')
            outf.write(f'{candidates}\n')
            outf.write(f'{scores}\n')
        metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
            metrics.append(f1)
        with open(args.out, 'a') as outf:
            outf.write(f'{metrics}\n')

    print("DONE!")
