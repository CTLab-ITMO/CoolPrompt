from readline import append_history_file
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import argparse
from utils.nat_inst_gpt3 import *
from sklearn.metrics import balanced_accuracy_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import entropy
import json
import heapq
import logging
import re

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--mode', default="Instruction Only", help='Mode of instructions/prompts')
parser.add_argument('--num-shots', default=2, type=int, help='Number of examples in the prompt if applicable')
parser.add_argument('--batch-size', default=4, type=int, help='Batch size')
parser.add_argument('--task-idx', default=1, type=int, help='The index of the task based on the array in the code')
parser.add_argument('--seed', default=42, type=int, help='Seed that changes sampling of examples')
parser.add_argument('--train-seed', type=int, help='Seed that changes the sampling of edit operations (search seed)')
parser.add_argument('--num-compose', default=1, type=int, help='Number of edits composed to get one candidate')
parser.add_argument('--num-train', default=100, type=int, help='Number of examples in score set')
parser.add_argument('--level', default="phrase", help='Level at which edit operations occur')
parser.add_argument('--simulated-anneal', action='store_true', default=False, help='Runs simulated anneal if candidate scores <= base score')
# parser.add_argument_('--GA', action='store_true', default=False, help='runs genetic algorithm instead of the greed strategy')
parser.add_argument('--agnostic', action='store_true', default=False, help='Uses template task-agnostic instruction')
parser.add_argument('--print-orig', action='store_true', default=False, help='Print original instruction and evaluate its performance')
parser.add_argument('--write-preds', action='store_true', default=False, help='Store predictions in a .json file')
parser.add_argument('--meta-dir', default='logs/', help='Path to store metadata of search')
parser.add_argument('--meta-name', default='search.txt', help='Path to the file that stores metadata of search')
parser.add_argument('--patience', default=2, type=int, help='The max patience P (counter)')
parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates in each iteration (m)')
parser.add_argument('--num-iter', default=10, type=int, help='Max number of search iterations')
parser.add_argument('--key-id', default=None, type=int, help='Use if you have access to multiple Open AI keys')
parser.add_argument('--edits', nargs="+", default=['del', 'swap', 'sub', 'add'], help='Space of edit ops to be considered')
parser.add_argument('--tournament-selection', default=2, type=int, help='Number of tournament selections for Genetic Algorithm')
parser.add_argument('--population-size', default=10, type=int, help='Population size for Genetic Algorithm')
parser.add_argument('--num-offspring', default=0, type=int, help='Number of the offspring for Genetic Algorithm')
parser.add_argument('--mutation-prob', default=0.5, type=float, help='Mutation probability for Genetic Algorithm')
parser.add_argument('--data-dir', default='./natural-instructions/tasks/', help='Path to the dataset')
parser.add_argument('--project-name', default='evolutional-prompt', help='Name of the wandb project')
parser.add_argument('--budget', default=1000, type=int, help='number of the budget of api calls for searching')
args = parser.parse_args()

try:
    import wandb
    wandb.login(key='c9039a6663b7aa3fa1260f5c004c21cce2584bd1')
    wandb.init(project=args.project_name)
    wandb.config.update(args)

except Exception as e:
    logger.warning("W&B logger is not available, please install to get proper logging")
    logger.error(e)

if args.key_id:
    import utils.nat_inst_gpt3
    utils.nat_inst_gpt3.key = args.key_id

meta_path = os.path.join(args.meta_dir, args.meta_name)
meta_file = open(meta_path, 'w+')
batch_size = args.batch_size
num_shots = args.num_shots
mode = args.mode
seed = args.seed
train_seed = args.train_seed

classification_task_ids = ['019', '021', '022', '050', '069', '137', '139','195']
data_base_path = args.data_dir #location of the Natural Instructions dataset
file_map = {f.split("_")[0]:f for f in os.listdir(data_base_path)}
assert args.task_idx >= 0 and args.task_idx < len(classification_task_ids), "Invalid task index entered."
chosen_task = classification_task_ids[args.task_idx] 
chosen_task_name = file_map['task' + chosen_task]
print("Running Experiment for: ", chosen_task_name)
file_contents = json.load(open("{}/{}".format(data_base_path, chosen_task_name)))
label_list = [file_contents["Instances"][i]["output"][0] for i in range(len(file_contents["Instances"])) ]
num_samples = 100 #default test set of size 100
num_train_samples = args.num_train

np.random.seed(train_seed)
torch.manual_seed(train_seed)
_, task_labels , _ = construct_instruction_prompt(mode='No Instructions', task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=seed, args=args)
task_labels = list(set(task_labels))
task_labels.sort()
print(task_labels)

instruction = file_contents['Definition']
print(instruction)
instruction[0].replace('\n' + 'Things to avoid: -', '')
print(instruction)
instruction = instruction[0].replace('\n' + 'Emphasis & Caution: -', '')
print(instruction)
if args.agnostic:
    instruction = "You will be given a task. Read and understand the task carefully, and appropriately answer '{}' or '{}'.".format(task_labels[0], task_labels[1])
parser = Parser.load('crf-con-en')
num_compose = args.num_compose
num_candidates = args.num_candidates
num_steps = args.num_iter
num_tournaments=args.tournament_selection
T_max = 10
edit_operations = args.edits
use_add = 'add' in edit_operations
population_size = args.population_size
num_offspring = args.num_offspring
mutation_prob = args.mutation_prob

wandb.log({"num_compose": num_compose})
wandb.log({"num_candidates": num_candidates})
wandb.log({"max_iter": num_steps})
wandb.log({"num_tournaments": num_tournaments})
wandb.log({"edit_operations": edit_operations})
wandb.log({"population_size": population_size})
wandb.log({"num_offspring": num_offspring})
wandb.log({"patience": args.patience})
wandb.log({"mutation_prob": mutation_prob})

if 'sub' in edit_operations:
    para_model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

def traverse_tree(parsed_tree):
    phrases = []
    for tree in parsed_tree:
        if tree.label() == '_': continue
        phrases.append(detokenize(tree.leaves()))
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_': continue
                phrases.append(detokenize(subtree.leaves()))
                phrases.extend(traverse_tree(subtree))
    return phrases

def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_': 
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves

def get_phrases(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

#phrases with punctuations
def get_phrases_pun(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases]
    return phrases

def get_response(input_text,num_return_sequences,num_beams):
  batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
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
    if phrase_1 in phrase_2:
        if candidate.find(' ' + phrase_2 + ' ') >= 0 : 
            candidate = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
        else: candidate = candidate.replace(phrase_2, '<2>')
        answer = candidate
        if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
            answer = answer.replace(' ' + phrase_1 + ' ', ' <1> ')
        else: answer = answer.replace(phrase_1, '<1>')        
        answer = answer.replace('<1>', phrase_2)
        answer = answer.replace('<2>', phrase_1)
    else:
        if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
            candidate = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
        else: candidate = candidate.replace(phrase_1, '<1>')
        answer = candidate
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
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
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

def custom_instruction_prompt(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=None, null_word=None, split='train', modified={},args=None):
    if mode=="Instruction Only":
        prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encode_instruction(task_name, instruction_structure = ["Definition"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, modified=modified, args=args)
    elif mode=="Instruction + Positive Examples":
        prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encode_instruction(task_name, instruction_structure = ["Definition", "Positive Examples Full Only"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, modified=modified, args=args)
    else: raise ValueError()
    if split == 'test': return prompt_list, answer_list, index_list
    elif split == 'train': 
        train_prompt_list.extend(dev_prompt_list)
        train_answer_list.extend(dev_answer_list)
        train_index_list.extend(dev_index_list)
        try:
            random.seed(data_seed)
            indices = random.sample(range(len(train_index_list)), num_train_samples) 
            train_prompt_list = [train_prompt_list[i] for i in indices]
            train_answer_list = [train_answer_list[i] for i in indices]
            train_index_list = [train_index_list[i] for i in indices]
        except: pass
        
        return train_prompt_list, train_answer_list, train_index_list

    else: raise ValueError()

def score(candidate, split='train', write=False):
    
    label_probs, calibrated_label_probs , raw_acc_count , raw_cal_acc_count , answer_list, index_list, _ = run(mode=mode, batch_size=batch_size, num_shots=num_shots, chosen_task_name=chosen_task_name, num_samples=num_samples, data_seed=seed, override_prompts=True, function = custom_instruction_prompt, split=split, modified={'Definition': candidate}, task_labels=task_labels, if_calibrate = False, args=args)
    preds = get_prediction(label_probs, task_labels)
    raw_acc = balanced_accuracy_score(answer_list, preds)
    label_frequencies = [preds.count(l)/len(preds) for l in task_labels]
    if split == 'train': return np.round(100*raw_acc, 2) + 10*entropy(label_frequencies)
    elif split== 'test': 
        if write:
            pname = args.meta_name
            pname = pname.split('.')[0] + "_predictions.json"
            pred_dump = {'predictions': preds, 'answers': answer_list, 'ids':index_list}
            ppath = os.path.join(args.meta_dir, pname)
            pfile = open(ppath, 'w+')
            json.dump(pred_dump, pfile)
        return np.round(100*raw_acc_count/len(answer_list), 2)
    else: return

def get_phrase_lookup(base_candidate):# with punctuation
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


def get_phrase_lookup_pun(base_candidate):
    if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(get_phrases_pun(base_candidate))}
    elif args.level == 'word': 
        words = word_tokenize(base_candidate)
        words = [w for w in words]
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

def tournament_selection(population, population_scores, num_tournaments):
    S_candidates = []
    S_scroes = []
    for k in range(num_tournaments):  
        parent = np.random.randint(0,len(population))  # parent, score_parent <-- Random(W)
        S_candidates.append(population[parent])  # S_candidate = S_candidate + parent 
        S_scroes.append(population_scores[parent])  # S_score = S_score + score_parent
    base_idx = np.argmax(S_scroes)   # base_idx = \arg max_{idx \in S} S_score
    base_candidate = S_candidates[base_idx] # base <-- S_candidates(base_idx)
    base_score = S_scroes[base_idx] # base_score <-- S_candidates(base_idx)
    return base_candidate, base_score    

def crossover(parent_1, parent_2):
    flag_error = False
    try:
        phrases_1_pun = get_phrase_lookup_pun(parent_1)
        phrases_2_pun = get_phrase_lookup_pun(parent_2)
    except AttributeError:
        offspring=''
        flag_error = True
        return offspring, flag_error
    phrases_1 = [phrase for phrase in list(phrases_1_pun.values()) if phrase not in string.punctuation or phrase == '']
    phrases_2 = [phrase for phrase in list(phrases_2_pun.values()) if phrase not in string.punctuation or phrase == '']

    split = np.random.randint(0,max(len(phrases_1),len(phrases_2)))
    
    if split >= len(phrases_1):
        offspring_phrases = list(phrases_1_pun.values()) + list(phrases_2_pun.values())[list(phrases_2_pun.values()).index(phrases_2[split]):]
    elif split >= len(phrases_2):
        offspring_phrases = list(phrases_1_pun.values())[0:list(phrases_1_pun.values()).index(phrases_1[split])]
    else:
        offspring_phrases = list(phrases_1_pun.values())[0:list(phrases_1_pun.values()).index(phrases_1[split])] + list(phrases_2_pun.values())[list(phrases_2_pun.values()).index(phrases_2[split]):]
    offspring_words = []
    for phrase in offspring_phrases:
        offspring_words = offspring_words + word_tokenize(phrase)
    offspring = detokenize(offspring_words)
    return offspring, flag_error

def update_result(patience_counter, result_candidate, result_score, population, population_scores):
    best_idx = np.argmax(population_scores) # k <-- \argmax_{j} s[j]
    best_score = population_scores[best_idx] # s_best <-- s[k]
    best_candidate = population[best_idx] # best <-- C[k]
    if best_score > result_score: 
        patience_counter = 1
        result_candidate = best_candidate
        result_score = best_score
    else: patience_counter += 1
    return patience_counter, result_candidate, result_score

def containenglish(str0):
    return bool(re.search('[a-z A-Z]', str0))
        
W_candidates = [] 
W_scores = []
W_deletesets = []
original_candidate = detokenize(word_tokenize(instruction))
assert word_tokenize(original_candidate) == word_tokenize(instruction)
meta_file.write("Original Candidate:\t "+ original_candidate + '\n')
original_score = score(original_candidate)
meta_file.write("Original Score:\t "+ str(original_score) + '\n')
meta_file.write("\n")

wandb.log({"original_score": original_score})

#Initialize population
W_candidates.append(original_candidate) # W_candidate <-- original_candidate
W_scores.append(original_score)  # W_scores <-- original_score
result_candidate = original_candidate # result_candidate <-- base candidate
result_score = original_score # result_score <-- base score
# W_0 <-- {<base , score_base> \times population_size}
W_candidates = W_candidates * population_size 
W_scores = W_scores * population_size
for i in range(population_size):
    W_deletesets.append([])   # record a DeleteSet(D) for each prompt in the population


patience_counter = 1

for i in range(num_steps):
    meta_file.write("Running step:\t " + str(i) + '\n')
    
    wandb.log({"step": i})
    
    # if i > 5:
    #     num_offspring = args.num_offspring
    # else: num_offspring = 0

    for j in range(num_offspring):

        parent_1, parent_score_1 = tournament_selection(W_candidates, W_scores, num_tournaments) # parent_1, parent_score_1 <-- tournament(W_{i-1})
        parent_2, parent_score_2 = tournament_selection(W_candidates, W_scores, num_tournaments) # parent_1, parent_score_1 <-- tournament(W_{i-1})
        meta_file.write("parent_1" + str(j) + ":\t " + parent_1+ '\n')
        meta_file.write("parent_2" + str(j) + ":\t " + parent_2+ '\n')

        #################### zhushi
        empty = True

        while empty:

            offspring, flag_error = crossover(parent_1, parent_2) # offspring <-- crossover(parent_1, parent_2)

            empty = not containenglish(offspring) #### 

            if flag_error:
                break
        ####################

        
        if flag_error: 
            meta_file.write("AttributeError occurs (parser) and skip this crossover operation"+ '\n')
            print('AttributeError occurs (parser) and skip this crossover operation')
            continue
        
        meta_file.write("Offspring" + str(j) + ":\t " + offspring + '\n')
        if offspring not in W_candidates:
            offspring_score = score(offspring)
            # population W  <-- offspring
            W_candidates.append(offspring) 
            W_scores.append(offspring_score) 
            W_deletesets.append(W_deletesets[W_candidates.index(parent_1)])
            meta_file.write("Offspring:\t" + offspring + '\n') 
            meta_file.write("Adding offspring in the population" + '\n') 

    if num_offspring > 0:
        top_N_p_idx_list = heapq.nlargest(population_size, range(len(W_scores)), W_scores.__getitem__)
        # population W <-- Top-N_p(population W)
        W_candidates_top_N_p = [W_candidates[i] for i in top_N_p_idx_list]
        W_scores_top_N_p = [W_scores[i] for i in top_N_p_idx_list]
        W_deletesets_N_p = [W_deletesets[i] for i in top_N_p_idx_list]
        W_candidates = W_candidates_top_N_p
        W_scores = W_scores_top_N_p
        W_deletesets = W_deletesets_N_p
        patience_counter, result_candidate, result_score = update_result(patience_counter, result_candidate, result_score, W_candidates, W_scores) # update result
        
        wandb.log({"result_score_after_crossover": result_score})

    if patience_counter > args.patience:
        print('Ran out of patience')
        meta_file.write('Ran out of patience \n')
        break
    else : pass

    # W' <-- W
 
    
    W_candidates_m = W_candidates
    W_scores_m = W_scores
    for i, base_candidate in enumerate(W_candidates_m):
        if mutation_prob > np.random.random():            
            try:
                phrase_lookup = get_phrase_lookup(base_candidate)
            except AttributeError:
                W_scores.remove(W_scores[W_candidates.index(base_candidate)]) 
                W_candidates.remove(base_candidate)
                meta_file.write("AttributeError occurs (parser) and skip this mutation"+ '\n')
                print('AttributeError occurs (parser) and skip this mutation')
                continue

            if base_candidate == original_candidate:
                for p in phrase_lookup.values(): print(p)
            if use_add: 
                if len(W_deletesets[i]): 
                    if 'add' not in edit_operations: edit_operations.append('add')
                else: 
                    if 'add' in edit_operations: edit_operations.remove('add')

            #####################################################################        
            empty = True
            while empty:
                if num_compose == 1:
                    edits = np.random.choice(edit_operations, num_candidates)
                else: 
                    edits = []
                    for n in range(num_candidates):
                        edits.append(np.random.choice(edit_operations, num_compose))
                print(edits)

                # generate mutated candidates
                candidates = []
                for edit in edits:
                    if isinstance(edit, str): 
                        meta_file.write("Performing edit for mutation:\t "+ edit + '\n')
                        candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, W_deletesets[i])

                        empty = not containenglish(candidate)
                        
                        if not empty:
                            print(candidate)
                            meta_file.write("Candidate after mutation:\t "+ candidate + '\n')
                            candidates.append(candidate)
                            deleteset = []
                            if edit  == 'del': 
                                deleteset = W_deletesets[i] + [phrase_lookup[indices[0]]]
                            if edit == 'add': 
                                if len(indices): 
                                    deleteset = W_deletesets[i]
                                    deleteset.remove(indices[0])
                            W_deletesets.append(deleteset)
                    else:
                        meta_file.write(("Performing edit for mutation:\t "+ ' '.join(edit))+ '\n')
                        old_candidate = base_candidate
                        composed_deletes = []
                        composed_adds = []
                        for op in edit:
                            phrase_lookup = get_phrase_lookup(old_candidate)
                            new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup, W_deletesets[i])

                            empty = not containenglish(new_candidate)

                            if not empty:
                                print(new_candidate)
                                if op  == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                                if op == 'add': 
                                    if len(indices): composed_adds.append(indices[0])
                                old_candidate = new_candidate
                            else:
                                break

                        if not empty:
                            meta_file.write("Candidate after mutation:\t "+ new_candidate+ '\n')
                            candidates.append(new_candidate)
                            deleteset = []
                            if 'del' in edit: 
                                deleteset = W_deletesets[i] + composed_deletes
                            if 'add' in edit and len(composed_adds) > 0:
                                deleteset = W_deletesets[i]
                                for phrase in composed_adds:
                                    deleteset.remove(phrase) 
                            W_deletesets.append(deleteset)
                        
            #####################################################################    
            scores = []
            for c, candidate in enumerate(candidates):
                s = score(candidate)
                scores.append(s)
                meta_file.write("Score of the mutation:\t "+ str(s) + '\n')

            # W' <-- W' + {<mutated, mutated_score>}
            W_scores_m = W_scores_m + scores
            W_candidates_m = W_candidates_m + candidates
            W_deletesets_m = W_deletesets

    top_N_p_idx_list_m = heapq.nlargest(population_size, range(len(W_scores_m)), W_scores_m.__getitem__)
    W_candidates_m_top_N_p = [W_candidates_m[i] for i in top_N_p_idx_list_m]
    W_scores_m_top_N_p = [W_scores_m[i] for i in top_N_p_idx_list_m]
    W_deletesets_m_top_N_p = [W_deletesets_m[i] for i in top_N_p_idx_list_m]

    W_candidates = W_candidates_m_top_N_p
    W_scores = W_scores_m_top_N_p
    W_deletesets = W_deletesets_m_top_N_p

    for j in range(len(W_candidates)):
        meta_file.write("Population candidate"+ str(j)+ ":\t "+ W_candidates[j] + '\n')
        meta_file.write("Score of population candidate"+ str(j)+ ":\t " + str(W_scores[j])+ '\n')
        # meta_file.write("\n")
        # if W_candidates[j] in added.keys():
        #     print('Notice! Prev tracker: ', delete_tracker)
        #     for chunk in added[W_candidates[j]]: 
        #         try: delete_tracker.remove(chunk)
        #         except: pass
        #     print('Notice! New tracker: ', delete_tracker)
        # if W_candidates[j] in deleted.keys():
        #     delete_tracker.extend(deleted[W_candidates[j]])

    patience_counter, result_candidate, result_score = update_result(patience_counter, result_candidate, result_score, W_candidates, W_scores)
    
    wandb.log({"result_score_after_mutation": result_score})
    
    count = complete_gpt3.count

    if patience_counter > args.patience:
        print('Ran out of patience')
        meta_file.write('Ran out of patience \n')
        break
    elif count >= args.budget:
        print('Ran out of budget')
        break
    else : pass

    result_candidate = detokenize(word_tokenize(result_candidate))

wandb.log({"result_score": result_score})   

meta_file.write('\n')
print('APICalls for search:\t', complete_gpt3.count)
meta_file.write('APICalls for search:\t'+ str(complete_gpt3.count) + '\n')
meta_file.write('\n')

wandb.log({"apicalls_search": complete_gpt3.count})


print('\nTesting .... ')
meta_file.write('Testing .... \n')
if args.print_orig:
    print('Task:\t', chosen_task_name)
    print('Original Instruction:\t', original_candidate)
    orig_score = score(original_candidate, 'test')
    print('Original Accuracy:\t', str(orig_score))
    meta_file.write('Original Accuracy:\t'+ str(orig_score)+ '\n')

if result_candidate == original_candidate: 
    print('No viable candidate found!')
    meta_file.write('No viable candidate found!\n')
    print('APICalls:\t', complete_gpt3.count)
    meta_file.write('APICalls:\t'+ str(complete_gpt3.count) + '\n')
    wandb.log({"Original Accuracy": orig_score})
    exit()
searched_score = score(result_candidate, 'test', write=args.write_preds)

wandb.log({"searched_accuracy": searched_score})
wandb.log({"apicalls_total": complete_gpt3.count})

print('Accuracy after search:\t', str(searched_score))
print('Instruction after search:\t', result_candidate)
meta_file.write('Instruction after search:\t'+ result_candidate+ '\n')
meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
print('APICalls:\t', complete_gpt3.count)
meta_file.write('APICalls:\t'+ str(complete_gpt3.count) + '\n')
print('APICalls:\t', complete_gpt3.count)
meta_file.write('APICalls:\t'+ str(complete_gpt3.count) + '\n')

wandb.save(meta_path)