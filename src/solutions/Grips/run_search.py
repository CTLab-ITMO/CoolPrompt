import os
import sys

import time

project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)

# os.environ['TOKENIZERS_PARALLELISM'] = "false"

from src.data.base.datasets.multi_task_dataset import BaseMultiTaskDataset
from src.data.base.datasets.classification_dataset import BaseClassificationDataset
from src.data.base.datasets.dataset import BaseDataset

import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import numpy as np
import argparse

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from src.solutions.Grips.tree import collect_leaves, detokenize
from utils import setup_tokenizer, setup_vllm_model


from src.data.classification import MNLIDataset, MRDataset, QNLIDataset, SST2Dataset, TrecDataset, YahooDataset
from src.data.generation import GSM8KDataset, MathDataset, SamsumDataset
from src.data.multi_task import BBHDataset, NaturalInstructionsDataset
from src.data.qa import MedQADataset, OpenbookQADataset

from src.evaluation.evaluator import BaseNLPEvaluator, TextClassificationEvaluator, GenerationEvaluator



def get_phrases(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0] # type: ignore
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

def get_response(input_text,num_return_sequences,num_beams):
  batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=30, return_tensors="pt").to(torch_device)
  translated = para_model.generate(**batch,max_length=30,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=0.0)
  tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else: 
        answer = candidate.replace(phrase, '')
    return answer

def add_phrase(candidate, phrase, after):
    if after == '': answer = phrase + ' ' + candidate
    else: 
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else: 
            answer = candidate.replace(after, after + phrase )
    return answer

def swap_phrases(candidate, phrase_1, phrase_2):
    if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else: answer = candidate.replace(phrase_1, '<1>')
    if candidate.find(' ' + phrase_2 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
    else: answer = candidate.replace(phrase_2, '<2>')
    answer = answer.replace('<1>', phrase_2)
    answer = answer.replace('<2>', phrase_1)
    return answer

def substitute_phrase(candidate, phrase):
    num_beams = 10
    num_return_sequences = 10
    paraphrases = get_response(phrase, num_return_sequences, num_beams)
    paraphrase = np.random.choice(paraphrases, 1)[0] 
    paraphrase = paraphrase.strip('.')
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', paraphrase + ' ')
    else: 
        answer = candidate.replace(phrase, paraphrase)
    return answer

def perform_edit(edit, base, phrase_lookup, delete_tracker):
    if edit == 'del':
        keys = list(phrase_lookup.keys())
        if len(keys) == 0:
            return base, [0]
        [i] = np.random.choice(keys, 1) 
        return delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False) 
        except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True) 
        return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        return substitute_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1) 
        if i >= 0: after = phrase_lookup[i]
        else: after = ''
        if len(delete_tracker) == 0: return base, []
        phrase = np.random.choice(delete_tracker, 1)[0]
        return add_phrase(base, phrase, after), [phrase]



# TASKS = [
#     "gsm8k",
#     "math",
#     "medqa",
#     "mnli",
#     "mr",
#     "natural_instructions_021",
#     "natural_instructions_050",
#     "natural_instructions_069",
#     "openbookqa",
#     "qnli",
#     "samsum",
#     "sst-2",
#     "trec",
#     "yahoo",
#     "bbh_boolean_expressions",
#     "bbh_hyperbaton",
#     "bbh_temporal_sequences",
#     "bbh_object_counting",
#     "bbh_disambiguation_qa",
#     "bbh_logical_deduction_three_objects",
#     "bbh_logical_deduction_five_objects",
#     "bbh_logical_deduction_seven_objects",
#     "bbh_causal_judgement",
#     "bbh_date_understanding",
#     "bbh_ruin_names",
#     "bbh_word_sorting",
#     "bbh_geometric_shapes",
#     "bbh_movie_recommendation",
#     "bbh_salient_translation_error_detection",
#     "bbh_formal_fallacies",
#     "bbh_penguins_in_a_table",
#     "bbh_dyck_languages",
#     "bbh_multistep_arithmetic_two",
#     "bbh_navigate",
#     "bbh_reasoning_about_colored_objects",
#     "bbh_tracking_shuffled_objects_three_objects",
#     "bbh_tracking_shuffled_objects_five_objects",
#     "bbh_tracking_shuffled_objects_seven_objects",
#     "bbh_sports_understanding",
#     "bbh_snarks",
#     "bbh_web_of_lies",
# ]

TASK_TO_DS = {
    
    "natural_instructions/task021": NaturalInstructionsDataset,
    "natural_instructions/task050": NaturalInstructionsDataset,
    "natural_instructions/task069": NaturalInstructionsDataset,
    
    # "math": MathDataset,
    # "sst-2": SST2Dataset,
    # "gsm8k": GSM8KDataset,

    # "yahoo": YahooDataset,
    # "trec": TrecDataset,
    # "mr": MRDataset,
    
    # "openbookqa": OpenbookQADataset,
    # "samsum": SamsumDataset,
    
    # "qnli": QNLIDataset,
    # "mnli": MNLIDataset,
    # "medqa": MedQADataset,
    
        
    "bbh/boolean_expressions" : BBHDataset,
    "bbh/hyperbaton" : BBHDataset,
    "bbh/temporal_sequences" : BBHDataset,
    "bbh/object_counting" : BBHDataset,
    "bbh/disambiguation_qa" : BBHDataset,
    "bbh/logical_deduction_three_objects" : BBHDataset,
    "bbh/logical_deduction_five_objects" : BBHDataset,
    "bbh/logical_deduction_seven_objects" : BBHDataset,
    "bbh/causal_judgement" : BBHDataset,
    "bbh/date_understanding" : BBHDataset,
    "bbh/ruin_names" : BBHDataset,
    "bbh/word_sorting" : BBHDataset,
    "bbh/geometric_shapes" : BBHDataset,
    "bbh/movie_recommendation" : BBHDataset,
    "bbh/salient_translation_error_detection" : BBHDataset,
    "bbh/formal_fallacies" : BBHDataset,
    "bbh/penguins_in_a_table" : BBHDataset,
    "bbh/dyck_languages" : BBHDataset,
    "bbh/multistep_arithmetic_two" : BBHDataset,
    "bbh/navigate" : BBHDataset,
    "bbh/reasoning_about_colored_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_three_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_five_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_seven_objects" : BBHDataset,
    "bbh/sports_understanding" : BBHDataset,
    "bbh/snarks" : BBHDataset,
    "bbh/web_of_lies" : BBHDataset
}

def extract_multitask_name(task_name: str) -> str:
    return task_name.split('/')[-1]


def create_ds_from_task(task_name: str, **kwargs) -> BaseDataset:
    ds_base = TASK_TO_DS[task_name](**kwargs)

    if isinstance(ds_base, BaseMultiTaskDataset):
        return ds_base.task(extract_multitask_name(task_name))

    return ds_base


def get_task_optimization_metric(ds: BaseDataset) -> str:
    
    if isinstance(ds, BaseClassificationDataset):
        return "f1"
    
    return "meteor"


def get_task_evaluator(ds: BaseDataset) -> BaseNLPEvaluator:
    
    if isinstance(ds, BaseClassificationDataset):
        return TextClassificationEvaluator()
    
    return GenerationEvaluator()


class Scorer:

    def __init__(self, model, tokenizer, task, evaluator, sample=100):
        self.model = model
        self.tokenizer= tokenizer
        self.ds_cls = TASK_TO_DS[task]
        self.task_name = task
        
        self.evaluator = evaluator
        
        self.sample = sample

    def score(self, prompt, split='train'):
        
        eval_ds = create_ds_from_task(
            self.task_name,
            tokenizer=self.tokenizer,
            split=split,
            prompt=prompt,
            sample=self.sample
        )
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    

        return self.evaluator.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=eval_ds,
            batch_size=args.batch_size,
            model_generate_args = {
                "stop_token_ids": terminators,
            }
        )



def get_phrase_lookup(base_candidate):
    if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(get_phrases(base_candidate))}
    elif args.level == 'word': 
        words = word_tokenize(base_candidate)
        words = [w for w in words if w not in string.punctuation or w != '']
        phrase_lookup = {p:phrase for p, phrase in enumerate(words)}
    elif args.level == 'sentence':
        sentences = sent_tokenize(base_candidate)
        phrase_lookup = {p:phrase for p, phrase in enumerate(sentences)}
    elif args.level == 'span':
        phrases = []
        for sentence in sent_tokenize(base_candidate):
            spans_per_sentence = np.random.choice(range(2,5)) # split sentence into 2, 3, 4, 5 chunks
            spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
            spans = [detokenize(s) for s in spans]
            phrases.extend(spans)
        phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
    else: raise ValueError()
    return phrase_lookup




def solve_task(task: str):
    
    start_time = time.time()
    
    
    task_ds_example =  create_ds_from_task(task, tokenizer=tokenizer, sample=1) 


    instruction = task_ds_example._get_basic_prompt()

    
    operations_tracker = []
    base_candidate = detokenize(word_tokenize(instruction))

    #assert word_tokenize(base_candidate) == word_tokenize(instruction), (base_candidate, instruction)
    original_candidate = instruction
    
    evaluator = get_task_evaluator(task_ds_example)

    metric_name = get_task_optimization_metric(task_ds_example)

    scorer = Scorer(
        model=model,
        tokenizer=tokenizer,
        task=task,
        evaluator=evaluator,
        sample=100
    )

    meta_file.write("Base Candidate:\t " + original_candidate + '\n')
    print("Base Candidate:\t " + original_candidate + '\n')

    base_score = scorer.score(base_candidate)[metric_name]

    print("Base scores", base_score)
    meta_file.write("Base Score:\t " + str(base_score) + '\n')
    meta_file.write("\n")

    delete_tracker = []
    patience_counter = 1

    num_steps = args.num_iter
    for i in range(num_steps):
        meta_file.write("Running step:\t " + str(i) + '\n')
        deleted = {}
        added = {}
        phrase_lookup = get_phrase_lookup(base_candidate)

        if base_candidate == original_candidate:
            for p in phrase_lookup.values(): print(p)

        if use_add:
            if len(delete_tracker):
                if 'add' not in edit_operations: edit_operations.append('add')
            else:
                if 'add' in edit_operations: edit_operations.remove('add')

        if num_compose == 1:
            edits = np.random.choice(edit_operations, num_candidates)
        else:
            edits = []
            for n in range(num_candidates):
                edits.append(np.random.choice(edit_operations, num_compose))

        print(edits)

        # generate candidates
        candidates = []
        for edit in edits:
            if isinstance(edit, str):
                meta_file.write("Performing edit:\t " + edit + '\n')
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker) # type: ignore
                meta_file.write("Generated candidate:\t " + candidate + '\n')
                candidates.append(candidate)
                if edit == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == 'add':
                    if len(indices): added[candidate] = indices
            else:
                meta_file.write(("Performing edit:\t " + ' '.join(edit)) + '\n')
                old_candidate = base_candidate
                composed_deletes = []
                composed_adds = []
                for op in edit:
                    phrase_lookup = get_phrase_lookup(old_candidate)
                    new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup, delete_tracker) # type: ignore
                    if op == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                    if op == 'add':
                        if len(indices): composed_adds.append(indices[0])
                    old_candidate = new_candidate
                meta_file.write("Generated candidate:\t " + new_candidate + '\n') # type: ignore
                candidates.append(new_candidate) # type: ignore # type: ignore
                if 'del' in edit: deleted[new_candidate] = composed_deletes # type: ignore
                if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds # type: ignore

        candidates = list(set(candidates)) # dedup
        print(candidates)
        scores = []
        for c, candidate in enumerate(candidates):
            scores.append(
                scorer.score(candidate)[metric_name]
            )
            print(scores[-1])
            meta_file.write("Score for Candidate " + str(c) + ":\t " + str(scores[-1]) + '\n')

        meta_file.write("\n")
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        if best_score > base_score:
            patience_counter = 1
            base_candidate = candidates[best_idx]
            base_score = best_score
            operations_tracker.append(edits[best_idx])
            meta_file.write("New Candidate Found" + '\n')
            meta_file.write("New Candidate Index:\t " + str(best_idx) + '\n')
            meta_file.write("New Candidate:\t " + base_candidate + '\n')
            meta_file.write("New Candidate Score:\t " + str(base_score) + '\n')
            try:
                meta_file.write("New Candidate Edit:\t " + edits[best_idx] + '\n')
            except:
                meta_file.write("New Candidate Edit:\t " + ' '.join(edits[best_idx]) + '\n')
            meta_file.write("\n")
            print('New Base Candidate: ', base_candidate)
            if base_candidate in added.keys():
                print('Notice! Prev tracker: ', delete_tracker)
                for chunk in added[base_candidate]:
                    try:
                        delete_tracker.remove(chunk)
                    except:
                        pass
                print('Notice! New tracker: ', delete_tracker)
            if base_candidate in deleted.keys():
                delete_tracker.extend(deleted[base_candidate])
            base_candidate = detokenize(word_tokenize(base_candidate))

        else:
            patience_counter += 1

            if patience_counter > args.patience:
                print('Ran out of patience')
                meta_file.write('Ran out of patience \n')
                break
            else:
                continue


    meta_file.write('\n')
    print('\nTesting .... ')
    meta_file.write('Testing .... \n')

    print('Task:\t', task)
    print('Original Instruction:\t', original_candidate)
    orig_score = scorer.score(original_candidate, split='test')
    for metric, value in orig_score.items():
        print(f'Original {metric}:\t', round(value, 5))
        meta_file.write(f'Original {metric}:\t' + str(round(value, 5)) + '\n')


    if base_candidate == original_candidate:
        print('No viable candidate found!')
        meta_file.write('No viable candidate found!\n')
        return

    searched_score = scorer.score(base_candidate, split='test')
    total_time = int(time.time() - start_time)
    meta_file.write('Instruction after search:\t' + base_candidate + '\n')
    for metric, value in searched_score.items():
        print(f'{metric} after search:\t', round(value, 5))
        meta_file.write(f'{metric} after search:\t' + str(round(value, 5)) + '\n')


    print('Instruction after search:\t', base_candidate)
    print('Edit Operations:\t', operations_tracker)
    print('Time taken:\t', total_time, "s")


    meta_file.write('Edit Operations:\t' + ' '.join([str(o) for o in operations_tracker]) + '\n')
    meta_file.write(f'Time taken: {total_time}s')

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--seed', type=int, help='Type in seed that changes sampling of examples')
    parser.add_argument('--num-compose', default=2, type=int, help='Number of edits composed to get one candidate')
    parser.add_argument('--level', default="word", help='level at which edit operations occur')
    parser.add_argument('--batch-size', type=int, default=100, help='Train / Test batch size')

    parser.add_argument('--print-orig', action='store_true', default=False,
                        help='print original instruction and evaluate its performance')
    parser.add_argument('--write-preds', action='store_true', default=False, help='store predictions in a .json file')
    parser.add_argument('--meta-dir', default='src/solutions/Grips/logs/', help='folder location to store metadata of search')
    parser.add_argument('--meta-name', default='search.txt', help='file name to store metadata of search')
    parser.add_argument('--patience', default=2, type=int, help='Type in the max patience P (counter)')
    parser.add_argument('--num-iter', default=7, type=int, help='Max number of search iterations')
    parser.add_argument('--edits', nargs="+", default=['sub', 'swap', 'del', 'add'],
                        help='space of edit ops to be considered')
    parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates in each iteration (m)')

    parser.add_argument('--task', default="sst-2", type=str, help='Task name')
    parser.add_argument('--task-dir', default="data", type=str, help='Task name')
    parser.add_argument('--model-name', default="AnatoliiPotapov/T-lite-instruct-0.1", type=str,
                        help='HF model full name')

    args = parser.parse_args()

    train_seed = 100
    
    tokenizer = setup_tokenizer(args.model_name)
    
    model = setup_vllm_model(args.model_name)
    
    num_samples = 100  # default test set of size 100

    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    
    parser = Parser.load('crf-con-en')
    num_compose = args.num_compose
    num_candidates = args.num_candidates
    num_steps = args.num_iter
    
    edit_operations = args.edits
    use_add = 'add' in edit_operations

    if 'sub' in edit_operations:
        para_model_name = 'tuner007/pegasus_paraphrase'
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
        para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval() # type: ignore

    
    for task in TASK_TO_DS.keys():
        
        print("----------------------------------------------")
        print("RUNNING Experiment for: ", task)
    
        meta_path = os.path.join(args.meta_dir, f'{task}.txt')
        meta_file = open(meta_path, 'w+')
        
        solve_task(task)
        
        print("FINISHED Experiment for: ", task)
        print("----------------------------------------------")


    