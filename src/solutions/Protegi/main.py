import os
import sys

project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)

from transformers import AutoTokenizer

import evaluators
from tqdm import tqdm
import time
import json
import argparse
import scorers
import optimizers

from src.data.classification import SST2Dataset
from src.evaluation.evaluator import TextClassificationEvaluator
from src.utils.eval_utils import Infer


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
    parser.add_argument('--n_test_exs', default=100, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="AnatoliiPotapov/T-lite-instruct-0.1", type=str)

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

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    default_model_generate_args = {
        "stop_token_ids": terminators
    }

    server_wrapper = Infer(model_name=args.engine)

    train_scorer = scorers.Cached01Scorer(
        args.engine,
        dataset_cls=TASK_TO_DS_MAP[args.task],
        tokenizer=tokenizer,
        split='train',
        default_gen_args=default_model_generate_args,
        ds_scorer=TASK_TO_SCORER_MAP[args.task](),
        sample=100
    )

    test_scorer = scorers.Cached01Scorer(
        args.engine,
        dataset_cls=TASK_TO_DS_MAP[args.task],
        tokenizer=tokenizer,
        split='test',
        default_gen_args=default_model_generate_args,
        ds_scorer=TASK_TO_SCORER_MAP[args.task](),
        sample=100
    )

    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, train_scorer, args.max_threads, bf_eval)

    ds_cls = TASK_TO_DS_MAP[args.task]

    base_prompt = ds_cls(
        tokenizer=tokenizer,
        split='train'
    )._get_basic_prompt()

    if os.path.exists(args.out):\
        
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    # Поддерживаем инвариант, что в них только чистый промпт, без формата.
    candidates = [base_prompt]

    epochs = config['rounds'] + 1

    for round in tqdm(range(epochs)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates)

        # score candidates
        scores = optimizer.score_candidates(candidates)
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

        metrics = test_scorer(candidates)
        with open(args.out, 'a') as outf:
            outf.write(f'{metrics}\n')

    print("DONE!")
