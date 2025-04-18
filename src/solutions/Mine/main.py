import os
import sys


project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)

from vllm import LLM

from transformers import AutoTokenizer




from tqdm import tqdm
import time
import argparse

from src.utils.eval_utils import TASK_TO_DS, LLMWrapper, create_ds_from_task, get_task_evaluator, get_task_optimization_metric
from src.solutions.Mine.utils import CachingEvaluator, seed_everyting
from src.solutions.Mine.generate import Candidate, PromptTransformer
from src.solutions.Mine.candidate import CandidateHistory
from src.solutions.Mine.sampler import TextSampler




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sst-2')
    parser.add_argument('--meta-dir', default='src/solutions/Mine/logs/', help='folder location to store metadata of search')


    parser.add_argument('--batch-size', type=int, default=100, help='Train and Test batch size')
    parser.add_argument('--epochs', type=int, default=6, help='Number of prompt optimization epochs')

    parser.add_argument('--model', default="AnatoliiPotapov/T-lite-instruct-0.1", type=str)

    args = parser.parse_args()

    return args



def solve_task(task: str):
    
    start_time = time.time()
    
    train_ds =  create_ds_from_task(task, tokenizer=tokenizer, split='train', sample=100)
    
    train_metric = get_task_optimization_metric(train_ds)
    
    caching_evaluator = CachingEvaluator(
        task=task,
        model=model,
        tokenizer=tokenizer,
        base_evaluator=get_task_evaluator(train_ds),
        default_gen_args=default_model_generate_args,
        batch_size=config['batch_size'],
    )
    
    base_prompt = train_ds._get_basic_prompt()
    base_score = caching_evaluator(base_prompt)[train_metric]
    
    base_candidate = Candidate(base_prompt, base_score)

    # Поддерживаем инвариант, что в них только чистый промпт, без формата.
    best_candidate = base_candidate

    epochs = config['epochs']
    
    sampler = TextSampler(train_ds)
    
    gen = PromptTransformer(wrapper, sampler)
    
    history = CandidateHistory()

    for round in tqdm(range(epochs)):
        print("STARTING ROUND ", round)
        start = time.time()
        
        history.clear()
        history.add(base_candidate)

        # 1. Generation
        gen_prompts = gen.generate_prompts(best_candidate)
        
        gen_candidates = [Candidate(p, caching_evaluator(p)[train_metric]) for p in gen_prompts]
        
        history.extend(gen_candidates)
        
        # 2. Distillation
        distilled_prompts = [gen.distill_samples(cand) for cand in gen_candidates]
        
        distilled_candidates = [Candidate(p, caching_evaluator(p)[train_metric]) for p in distilled_prompts]
        
        history.extend(distilled_candidates)
        
        # 3. Compression
        compressed_prompts = [gen.compress_prompt(cand) for cand in distilled_candidates]
        
        compressed_candidates = [Candidate(p, caching_evaluator(p)[train_metric]) for p in compressed_prompts]
        
        history.extend(compressed_candidates)

        
        # 4. Aggregation
        
        aggregated_prompt = gen.aggregate_prompts(compressed_candidates)

        aggregated_candidate = Candidate(aggregated_prompt, caching_evaluator(aggregated_prompt)[train_metric])
        
        aggregated_synonyms = gen.generate_synonyms(aggregated_candidate, n=3)
        
        final_candidates = [Candidate(p, caching_evaluator(p)[train_metric]) for p in aggregated_synonyms]
        final_candidates.append(aggregated_candidate)
        
        history.extend(final_candidates)

        best_candidate = history.get_highest_scorer()
        
        
        print("-----------------")
        print("Best candidate:")
    
        print(best_candidate.prompt)
        
        print("Train score:", best_candidate.train_score)
        print("Test score:", caching_evaluator(best_candidate.prompt, split='test'))


        meta_file.write(f"======== ROUND {round}\n")
        meta_file.write(f'{time.time() - start}\n')
        meta_file.write(f'Prompts: {[c.prompt for c in final_candidates]}\n')
        meta_file.write(f'Scores: {[c.train_score for c in final_candidates]}\n')


    print("DONE!")
    base_test_score = caching_evaluator(base_prompt, split='test')
    
    instr_fewshot = gen.convert_to_fewshot(base_candidate)
    instr_fewshot_score = caching_evaluator(instr_fewshot, split='test')
    
    clean_fewshot = gen.convert_to_fewshot(Candidate("", 0.0)).lstrip()
    clean_fewshot_score = caching_evaluator(clean_fewshot, split='test')
    
    best_prompt = best_candidate.prompt
    test_score = caching_evaluator(best_prompt, split='test')
    
    best_prompt_fewshot = gen.convert_to_fewshot(best_candidate)
    best_prompt_fewshot_score = caching_evaluator(best_prompt_fewshot, split='test')
    
    end_time = time.time() - start_time
    meta_file.write(f"Time taken: {end_time}s\n")
    
    meta_file.write(f"Base prompt: {base_prompt}\n")
    meta_file.write(f"Base prompt score: {base_test_score}\n")
    meta_file.write(f"-----------------------------\n")
    meta_file.write(f"Few shot prompt: {instr_fewshot}\n")
    meta_file.write(f"Few shot score: {instr_fewshot_score}\n")
    meta_file.write(f"-----------------------------\n")
    meta_file.write(f"Res prompt: {best_prompt}\n")
    meta_file.write(f"Res prompt score: {test_score}\n")
    meta_file.write(f"-----------------------------\n")
    meta_file.write(f"Clean fewshot prompt: {clean_fewshot}\n")
    meta_file.write(f"Clean fewshot score: {clean_fewshot_score}\n")
    meta_file.write(f"-----------------------------\n")
    meta_file.write(f"Res fewshot prompt: {best_prompt_fewshot}\n")
    meta_file.write(f"Res fewshot score: {best_prompt_fewshot_score}\n")

if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    model_name=config['model']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    default_model_generate_args = { # should only be passed into base evaluator calls
        "stop_token_ids": terminators
    }
    
    wrapper_gen_args = {
        "stop_token_ids": terminators,
        "max_tokens": 1024,
        "temperature": 0.15,
    }
    

    model = LLM(model=model_name, dtype="float16", trust_remote_code=True, gpu_memory_utilization=0.4)
    
    wrapper = LLMWrapper(model, wrapper_gen_args)
    
    
    seed = 100
    
    seed_everyting(seed)
    
    tasks = TASK_TO_DS.keys()
    
    #tasks = ['sst-2', 'gsm8k']

    for task in tasks:
        
        print("----------------------------------------------")
        print("RUNNING Experiment for: ", task)
    
        meta_path = os.path.join(config['meta_dir'], f'{task}.txt')
        
        os.makedirs(meta_path, exist_ok=True)
        
        meta_file = open(meta_path, 'w+')
        
        solve_task(task)
        
        print("FINISHED Experiment for: ", task)
        print("----------------------------------------------")

