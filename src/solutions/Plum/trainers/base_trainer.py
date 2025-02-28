import numpy as np
import torch
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
import utils.nat_inst_gpt3 as gpt3
import utils.nat_inst_gpt2 as gpt2
from utils.expanded_encode_instruction import *
from sklearn.metrics import balanced_accuracy_score
import json
import os
from scipy.stats import entropy
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pickle

class TrainerBase:
    
    """Base class for searching strategy."""

    def __init__(self, maxiter, patience, train_seed, data_seed, num_compose, num_candidates):
        super(TrainerBase, self).__init__()
        self.maxiter = maxiter
        self.patience = patience
        self.train_seed = train_seed
        self.data_seed = data_seed
        self.num_compose = num_compose
        self.num_candidates = num_candidates
        self.patience_counter = 1
        self.state = {}

    def choose_best(self, candidates, scores):
        best_idx = np.argmax(scores) # k <-- \argmax_{j} s[j]
        best_score = scores[best_idx] # s_best <-- s[k]
        best_candidate = candidates[best_idx] # best <-- C[k]
        return best_score, best_candidate

    def update_result(self, population, population_scores):
        best_score, best_candidate = self.choose_best(population, population_scores)
        if best_score > self.result_score: 
            self.patience_counter = 1
            self.result_candidate = best_candidate
            self.result_score = best_score
        else: self.patience_counter += 1

    @property
    def is_run_out_of_patience(self):
        return self.patience_counter > self.patience

    def save(self, save_dir):
        f = open(save_dir, 'wb')
        pickle.dump(self.state, f)
        f.close()

    def load(self, load_dir):
        f = open(load_dir, 'rb')
        self.state = pickle.load(f)
        f.close()
    
class SimpleTrainer(TrainerBase):

    """A simple class based on GrIPS."""

    def __init__(self, maxiter, patience, train_seed, data_seed, num_compose, num_candidates, backbone):
        super(SimpleTrainer, self).__init__(maxiter, patience, train_seed, data_seed, num_compose, num_candidates)
        if backbone == "gpt3":
            self.run = gpt3.run
            self.get_prediction = gpt3.get_prediction
        if backbone == "gpt2":
            self.run = gpt2.run
            self.get_prediction = gpt2.get_prediction
        self.patience_counter = 1
        self.W_candidates = []
        self.W_scores = []
        self.original_candidate = None
        self.original_score = None
        self.result_candidate = None
        self.result_score = None
        self.parser = Parser.load('crf-con-en')
        self.para_tokenizer = None
        self.para_model = None
        self.state = {}

    def detokenize(self, tokens):
        return TreebankWordDetokenizer().detokenize(tokens)

    def word_tokenize(slef, instruction):
        return word_tokenize(instruction)
    
    def sent_tokenize(slef, instruction):
        return sent_tokenize(instruction)

    def traverse_tree(self, parsed_tree):
        phrases = []
        for tree in parsed_tree:
            if tree.label() == '_': continue
            phrases.append(self.detokenize(tree.leaves()))
            for subtree in tree:
                if type(subtree) == nltk.tree.Tree:
                    if subtree.label() == '_': continue
                    phrases.append(self.detokenize(subtree.leaves()))
                    phrases.extend(self.traverse_tree(subtree))
        return phrases

    def check_child(self, tree):
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

    def collect_leaves(self, parsed_tree):
        leaves = []
        for tree in parsed_tree:
            if type(parsed_tree) != nltk.tree.Tree: continue
            if tree.label() == '_': 
                leaves.append(self.detokenize(tree.leaves()))
                continue
            if self.check_child(tree): leaves.append(self.detokenize(tree.leaves()))
            else:
                leaves.extend(self.collect_leaves(tree))
        return leaves

    def get_phrases(self, instruction): # one possible way of obtaining disjoint phrases
        phrases = []
        for sentence in self.sent_tokenize(instruction):
            parsed_tree = self.parser.predict(self.word_tokenize(sentence), verbose=False).sentences[0].trees[0]
            leaves = self.collect_leaves(parsed_tree)
            phrases.extend(leaves)
        phrases = [self.detokenize(self.word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
        return phrases

    def if_sub(self, edit_operations):
        if 'sub' in edit_operations:
            para_model_name = 'tuner007/pegasus_paraphrase'
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
            self.para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()

    def get_response(self, input_text, num_return_sequences, num_beams):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch = self.para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = self.para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def delete_phrase(self, candidate, phrase):
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ')
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', ' ')
        else: 
            answer = candidate.replace(phrase, '')
        return answer

    def add_phrase(self, candidate, phrase, after):
        if after == '': answer = phrase + ' ' + candidate
        else: 
            if candidate.find(' ' + after) > 0:
                answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
            elif candidate.find(after + ' ') > 0:
                answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
            else: 
                answer = candidate.replace(after, after + phrase )
        return answer
    
    def swap_phrases(self, candidate, phrase_1, phrase_2):
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

    def substitute_phrase(self, candidate, phrase):
        num_beams = 10
        num_return_sequences = 10
        paraphrases = self.get_response(phrase, num_return_sequences, num_beams)
        paraphrase = np.random.choice(paraphrases, 1)[0] 
        paraphrase = paraphrase.strip('.')
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', paraphrase + ' ')
        else: 
            answer = candidate.replace(phrase, paraphrase)
        return answer

    def perform_edit(self, edit, base, phrase_lookup, delete_tracker):
        if edit == 'del':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
            return self.delete_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'swap':
            try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False) 
            except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True) 
            return self.swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
        elif edit == 'sub':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
            return self.substitute_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'add':
            keys = list(phrase_lookup.keys())
            keys.append(-1)
            [i] = np.random.choice(keys, 1) 
            if i >= 0: after = phrase_lookup[i]
            else: after = ''
            if len(delete_tracker) == 0: return base, []
            phrase = np.random.choice(delete_tracker, 1)[0]
            return self.add_phrase(base, phrase, after), [phrase]

    def custom_instruction_prompt(self, mode=None, task_name=None, num_shots=None, num_test_instances=None, data_seed=None, null_word=None, split='train', modified={}, args=None):
        if mode=="Instruction Only":
            prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encode_instruction(task_name, instruction_structure = ["Definition"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, modified=modified, args=args)
        elif mode=="Instruction + Positive Examples":
            prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encode_instruction(task_name, instruction_structure = ["Definition", "Positive Examples Full Only"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, modified=modified, args=args)
        else: 
            raise ValueError()
        if split == 'test': 
            return prompt_list, answer_list, index_list
        elif split == 'train': 
            train_prompt_list.extend(dev_prompt_list)
            train_answer_list.extend(dev_answer_list)
            train_index_list.extend(dev_index_list)
            try:
                random.seed(data_seed)
                indices = random.sample(range(len(train_index_list)), args.num_samples) 
                train_prompt_list = [train_prompt_list[i] for i in indices]
                train_answer_list = [train_answer_list[i] for i in indices]
                train_index_list = [train_index_list[i] for i in indices]
            except: pass
            
            return train_prompt_list, train_answer_list, train_index_list

        else: raise ValueError()

    def score(self, candidate, split='train', write=False, args=None):
        task_labels = args.task_labels
        label_probs, calibrated_label_probs , raw_acc_count , raw_cal_acc_count , answer_list, index_list, _ = self.run(mode=args.mode, batch_size=args.batch_size, num_shots=args.num_shots, chosen_task_name=args.chosen_task_name, num_samples=args.num_samples, data_seed=args.data_seed, override_prompts=True, function = self.custom_instruction_prompt, split=split, modified={'Definition': candidate}, task_labels=task_labels, if_calibrate = False, args=args)
        preds = self.get_prediction(label_probs, task_labels)
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
        else: 
            return
        

    def get_phrase_lookup(self, base_candidate, args):
        if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(self.get_phrases(base_candidate))}
        elif args.level == 'word': 
            words = self.word_tokenize(base_candidate)
            words = [w for w in words if w not in string.punctuation or w != '']
            phrase_lookup = {p:phrase for p, phrase in enumerate(words)}
        elif args.level == 'sentence':
            sentences = self.sent_tokenize(base_candidate)
            phrase_lookup = {p:phrase for p, phrase in enumerate(sentences)}
        elif args.level == 'span':
            phrases = []
            for sentence in self.sent_tokenize(base_candidate):
                spans_per_sentence = np.random.choice(range(2,5)) # split sentence into 2, 3, 4, 5 chunks
                spans = np.array_split(self.word_tokenize(sentence), spans_per_sentence)
                spans = [self.detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
        else: raise ValueError()
        return phrase_lookup
    
    def get_phrases_pun(self, instruction): # one possible way of obtaining disjoint phrases
        phrases = []
        for sentence in sent_tokenize(instruction):
            parsed_tree = self.parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
            leaves = self.collect_leaves(parsed_tree)
            phrases.extend(leaves)
        phrases = [self.detokenize(word_tokenize(phrase)) for phrase in phrases]
        return phrases
    
    def get_phrase_lookup_pun(self, base_candidate, args):
        if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(self.get_phrases_pun(base_candidate))}
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
                spans = [self.detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
        else: raise ValueError()
        return phrase_lookup


    def init_population(self, instruction, args):  
        self.original_candidate = self.detokenize(self.word_tokenize(instruction))
        assert self.word_tokenize(self.original_candidate) == self.word_tokenize(instruction)
        # original_candidate = base_candidate
        self.original_score = self.score(self.original_candidate, args=args)
        self.W_candidates.append(self.original_candidate) # W_candidate <-- original_candidate
        self.W_scores.append(self.original_score)  # W_scores <-- original_score
        self.result_candidate = self.original_candidate # result_candidate <-- base candidate
        self.result_score = self.original_score # result_score <-- base score

    def update_result_add(self, best_score, best_candidate, use_simulated_anneal=False, i=None):
        add_best_or_not = False
        if best_score > self.result_score: 
            self.patience_counter = 1
            self.result_candidate = best_candidate
            self.result_score = best_score
            add_best_or_not = True
        else: 
            self.patience_counter += 1
            if use_simulated_anneal:
                K = 5
                T_max = 10
                T = T_max * np.exp(-i/K)
                prob = np.exp((best_score - self.result_score)/ T)
                if np.random.binomial(1, prob): 
                    self.result_candidate = best_candidate
                    self.result_score = best_score
                    add_best_or_not = True
        return add_best_or_not
    
    

