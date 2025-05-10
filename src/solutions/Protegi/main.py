import os
import sys


project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)

import numpy as np
import torch
from vllm import LLM

from transformers import AutoTokenizer

import evaluators
from tqdm import tqdm
import time
import json
import argparse
import scorers
import optimizers

from src.utils.eval_utils import TASK_TO_DS, LLMWrapper, create_ds_from_task, get_task_evaluator
from src.data.base.datasets.generation_dataset import BaseGenerationDataset


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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sst-2')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--meta-dir', default='src/solutions/Protegi/logs/', help='folder location to store metadata of search')
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=5, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=100, type=int)

    parser.add_argument('--batch-size', type=int, default=16, help='Train / Test batch size')
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


def solve_task(task: str):
    
    start_time = time.time()
    
    task_ds_example =  create_ds_from_task(task, tokenizer=tokenizer, sample=1)
    
    if isinstance(task_ds_example, BaseGenerationDataset):
        print("Skipping generation task") 
        return

    base_prompt = task_ds_example._get_basic_prompt()

    train_scorer = scorers.Cached01Scorer(
        model,
        task=task,
        tokenizer=tokenizer,
        split='train',
        default_gen_args=default_model_generate_args,
        ds_scorer=get_task_evaluator(task_ds_example),
        batch_size=args.batch_size,
        sample=100
    )

    test_scorer = scorers.Cached01Scorer(
        model,
        task=task,
        tokenizer=tokenizer,
        split='test',
        default_gen_args=default_model_generate_args,
        ds_scorer=get_task_evaluator(task_ds_example),
        batch_size=args.batch_size,
        sample=None
    )
    
    base_score = test_scorer([base_prompt])[0]

    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, train_scorer, wrapper, bf_eval)


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

        meta_file.write(f"======== ROUND {round}\n")
        meta_file.write(f'{time.time() - start}\n')
        meta_file.write(f'{candidates}\n')
        meta_file.write(f'{scores}\n')

        metrics = test_scorer(candidates)
        meta_file.write(f'{metrics}\n')

    print("DONE!")
    
    best_prompt = candidates[0]
    best_prompt_score = test_scorer([best_prompt])[0]
    
    end_time = time.time() - start_time
    meta_file.write(f"Time taken: {end_time}s\n")
    
    meta_file.write(f"Base prompt: {base_prompt}\n")

    meta_file.write(f"-----------------------------\n")
    meta_file.write(f"Res prompt: {best_prompt}\n")
    meta_file.write(f"-----------------------------\n")
    
    meta_file.write(f"Base prompt full score: {base_score}\n")
    meta_file.write(f"Res prompt full score: {best_prompt_score}\n")

if __name__ == '__main__':
    args = get_args()

    config = vars(args)


    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    print(config)

    
    tokenizer = AutoTokenizer.from_pretrained(args.engine)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    default_model_generate_args = {
        "stop_token_ids": terminators,
        "temperature": 0.0
    }
    
    wrapper_gen_args = {
        "stop_token_ids": terminators,
        "max_tokens": 1024,
        "temperature": 0.15,
    }
    
    model_name=args.engine
    
    model = LLM(model=model_name, dtype="float16", trust_remote_code=True, gpu_memory_utilization=0.4)
    
    wrapper = LLMWrapper(model, wrapper_gen_args)
    
    train_seed = 100
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    for task in TASK_TO_DS.keys():
        
        print("----------------------------------------------")
        print("RUNNING Experiment for: ", task)
    
        args.meta_dir = 'src/solutions/Protegi/logs/full_test/'
    
        meta_path = os.path.join(args.meta_dir, f'{task}.txt')

        dir_path = "/".join(meta_path.split("/")[:-1])
        
        os.makedirs(dir_path, exist_ok=True)
        
        meta_file = open(meta_path, 'w+')
        
        solve_task(task)
        
        print("FINISHED Experiment for: ", task)
        print("----------------------------------------------")

