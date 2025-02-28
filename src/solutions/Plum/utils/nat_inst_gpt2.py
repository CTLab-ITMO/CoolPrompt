import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import json
from utils.expanded_encode_instruction import *
from copy import deepcopy
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score as F1


# Extra define 
null_words = ["N/A", "", "[MASK]"]
num_top_tokens = 100
num_gen_tokens = 1
all_regular_preds = []
all_calibrated_preds = []
all_answers = []

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')


model.eval().cuda()
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
num_seeds = 5

def create_batches(test_instances, test_labels=[], batch_size=2):
    test_sentence_batches = []
    test_label_batches = []
    for i in range(0,len(test_instances),batch_size):
        test_sentence_batches.append(test_instances[i:i+batch_size])
        if len(test_labels) > 0: test_label_batches.append(test_labels[i: i + batch_size])
    if len(test_labels) > 0:
        return test_sentence_batches, test_label_batches
    else:
        return test_sentence_batches

def construct_instruction_prompt(mode, task_name, num_shots, num_test_instances, data_seed, null_word=None, args=None):
    if mode=="No Instructions" or mode==0:
        prompt_list, answer_list, index_list = encode_instruction(task_name, instruction_structure = [], number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, args=args)
    elif mode=="Instruction Only" or mode==1:
        prompt_list, answer_list, index_list = encode_instruction(task_name, instruction_structure = ["Definition"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, args=args)
    elif mode=="Instruction + Examples" or mode==2:
        prompt_list, answer_list, index_list = encode_instruction(task_name, instruction_structure = ["Definition", "Positive Examples Full Only"], number_of_examples = num_shots, number_of_instances = num_test_instances, data_seed=data_seed, null_word=null_word, args=args)
    else:
        raise ValueError("Invalid mode entry, mode not recognized")
    return prompt_list, answer_list, index_list

def token_task_labels(labels):
    token_list = []
    for l in labels:
        token_list.append(tokenizer.encode(l, return_tensors='pt'))
    return token_list

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
        return res
    wrapper.count = 0
    return wrapper

@counter
def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the context and the next l tokens
        logits = model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    del input_ids

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs 
                    curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

        choices.append(curr_json)
    return_json['choices'] = choices
    del total_sequences
    return return_json

def get_not_found_prob(prompt, label):
    prompt = prompt + label # space already in label
    label_ans = complete_gpt2(prompt, l=0, num_log_probs=1, echo=True)['choices'][0]
    return np.exp(label_ans['logprobs']['token_logprobs'][-1])


def get_regular_label_probs(response, batch, labels, if_null=False):
    # check if in top tokens
    assert len(response['choices']) == len(batch)
    label_probs = torch.zeros([len(response['choices']), 1, len(labels)])
    all_missing_positions = []
    for a, ans in enumerate(response['choices']):
        for l, label in enumerate(labels):
            if label in ans['logprobs']['tokens']:
                label_probs[a,:,l] = np.exp(ans['logprobs']['token_logprobs'][0])
            else:
                position = (a, l)
                all_missing_positions.append(position)
                
    if len(all_missing_positions) > 0:
        all_additional_prompts = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            missing_prompt = batch[which_sentence] + labels[which_label]
            all_additional_prompts.append(missing_prompt)
        additional_prompt_batches, position_lookup = create_batches(all_additional_prompts,all_missing_positions, batch_size=len(batch[0]))
        for m, missing_batch in enumerate(additional_prompt_batches):
            missing_response = complete_gpt2(missing_batch, l=0, num_log_probs=1, echo=True)
            for idx, missing_ans in enumerate(missing_response['choices']):
                which_sentence, which_label = position_lookup[m][idx]
                label_probs[which_sentence,:,which_label] = np.exp(missing_ans['logprobs']['token_logprobs'][-1])
    assert (label_probs > 0).all(), "all should be populated with non-zero value"
            
    if if_null: return label_probs
    label_probs = label_probs/torch.sum(label_probs, dim=2, keepdim=True)
    return label_probs 

def get_null_label_probs(null_batch, labels):
    null_label_probs = torch.zeros([len(null_batch), 1, len(labels)])
    for l, label in enumerate(labels):
        label_batch = [prompt + label for prompt in null_batch]
        label_response = complete_gpt2(label_batch, l=0, num_log_probs=1, echo=True)
        for a, ans in enumerate(label_response['choices']):
            null_label_probs[a,:,l] = np.exp(ans['logprobs']['token_logprobs'][-1])
    return null_label_probs

def get_prediction(label_probs, labels):
    pred_ids = torch.flatten(torch.argmax(label_probs, dim=2))
    preds = []
    for i in range(label_probs.shape[0]):
        preds.append(labels[pred_ids[i]])
    return preds

def evaluate_preds(batch_preds, batch_labels):
    batch_preds = np.array([p.strip(" ") for p in batch_preds])
    batch_labels = np.array(batch_labels)
    return batch_preds == batch_labels

def run(mode, batch_size, num_shots, chosen_task_name, num_samples, data_seed=0, logit_only=False, override_prompts=False, function=None, split=None, task_labels = [], modified = {}, if_calibrate = True, args=None):
    regular_accuracy_count = 0
    calibrated_accuracy_count = 0
    if not override_prompts: prompt_list, answer_list, index_list = construct_instruction_prompt(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=data_seed, args=args)
    else: prompt_list, answer_list, index_list = function(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=data_seed, split=split, modified=modified, args=args)
    prompt_batches, batch_test_labels = create_batches(prompt_list, answer_list, batch_size)
    if len(task_labels) == 0: task_labels = list(set(answer_list))
    task_labels = [" " + label for label in task_labels]
    task_labels.sort()
    print('Sanity Check, Task Labels are: ', task_labels)
    all_label_probs = []
    all_calibrated_probs = []
    temp_prompt_batches = []
    if if_calibrate:
        for nw in null_words:
            if not override_prompts: null_prompt_list, null_answer_list, null_index_list = construct_instruction_prompt(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=data_seed, null_word=nw)
            else: null_prompt_list, null_answer_list, null_index_list = function(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, data_seed=data_seed, null_word=nw, split=split, modified=modified)
            assert index_list == null_index_list, pdb.set_trace()
            null_batches, _ = create_batches(null_prompt_list, null_answer_list, batch_size)
            temp_prompt_batches.append(null_batches)
    null_prompt_batches = []
    if if_calibrate:
        for i in range(len(prompt_batches)):
            null_word_batches = []
            for j in range(len(null_words)):
                null_word_batches.append(temp_prompt_batches[j][i])
            null_prompt_batches.append(null_word_batches)
    

    all_batches = prompt_batches
    all_null_batches = null_prompt_batches
    
    for j in tqdm(range(len(all_batches))):
        batch = all_batches[j]
        
        responses = complete_gpt2(batch, l=num_gen_tokens, num_log_probs=num_top_tokens)
        label_probs = get_regular_label_probs(responses, batch, task_labels, if_null = logit_only)
        all_label_probs.append(label_probs)
        if logit_only: label_probs = label_probs/torch.sum(label_probs, dim=2, keepdim=True)

        regular_preds = get_prediction(label_probs, task_labels)

        regular_accuracy_count += np.sum(evaluate_preds(regular_preds, batch_test_labels[j]))
        all_regular_preds.extend([p.strip(" ") for p in regular_preds])

# perform calibration
        if if_calibrate:
            null_batches = all_null_batches[j]
            null_probs_list = []
            for null_batch in null_batches:
                null_probs = get_null_label_probs(null_batch, task_labels)
                null_probs_list.append(null_probs)


            null_probs = torch.mean(torch.stack(null_probs_list), dim=0)
            null_probs = null_probs/torch.sum(null_probs, dim=2, keepdim=True)
            num_classes = len(task_labels)

            calibrated_probs = label_probs/null_probs
            if logit_only: all_calibrated_probs.append(calibrated_probs)
            calibrated_probs = calibrated_probs/torch.sum(calibrated_probs, dim=2, keepdim=True)
            if not logit_only: all_calibrated_probs.append(calibrated_probs)
            calibrated_preds = get_prediction(calibrated_probs, task_labels)

            calibrated_accuracy_count += np.sum(evaluate_preds(calibrated_preds, batch_test_labels[j]))
            all_calibrated_preds.extend([p.strip(" ") for p in calibrated_preds])
        all_answers.extend(batch_test_labels[j])
    
        del batch, responses, regular_preds
        if if_calibrate: del null_batches, calibrated_preds
    
    all_label_probs = torch.cat(all_label_probs, dim=0)
    if if_calibrate: all_calibrated_probs = torch.cat(all_calibrated_probs, dim=0)
    return all_label_probs, all_calibrated_probs, regular_accuracy_count, calibrated_accuracy_count, answer_list, index_list, task_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--mode', default="Prompt Only", help='Type mode of instructions')
    parser.add_argument('--num-shots', default=2, type=int, help='Type number of examples in the prompt if applicable')
    parser.add_argument('--batch-size', default=4, type=int, help='Type in the batch-size')
    parser.add_argument('--task-idx', default=0, type=int, help='Type in the batch-size')
    parser.add_argument('--data-seed', default=0, type=int, help='Type in the batch-size')
    parser.add_argument('--save-preds', action='store_true')
    args = parser.parse_args()

    batch_size = args.batch_size
    num_shots = args.num_shots
    mode = args.mode

    classification_task_ids = ['019', '021', '022', '050', '069', '137', '139','195']
    data_base_path = "data/ExpandedNaturalInstructions/"
    file_map = {f.split("_")[0]:f for f in os.listdir(data_base_path)}
    assert args.task_idx >= 0 and args.task_idx < len(classification_task_ids), "Invalid task index entered."
    chosen_task = classification_task_ids[args.task_idx] 
    chosen_task_name = file_map['task' + chosen_task]
    print("Running Experiment for: ", chosen_task_name)
    file_contents = json.load(open("{}/{}".format(data_base_path, chosen_task_name)))
    num_samples = 100 
    dest_path = "preds/{}/{}/".format(chosen_task, args.mode)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    label_probs, calibrated_probs, regular_accuracy_count, calibrated_accuracy_count, answer_list, index_list, task_labels = run(mode=mode, batch_size=batch_size, num_shots=num_shots, chosen_task_name=chosen_task_name, num_samples=num_samples, data_seed=args.data_seed)
    print("Regular Accuracy:\t", np.round(100*regular_accuracy_count/len(answer_list), 2))
    print("Calibrated Accuracy:\t", np.round(100*calibrated_accuracy_count/len(answer_list), 2))
    max_label = max(set(answer_list), key=answer_list.count)
    print("Majority Baseline Accuracy:\t", np.round(100*answer_list.count(max_label)/len(answer_list),2))
    assert len(all_regular_preds) == len(all_calibrated_preds)
    assert len(all_regular_preds) == len(answer_list)
    assert all_answers == answer_list
    del all_answers
    print("Regular F1:\t", np.round(100*F1(answer_list, all_regular_preds, average='macro'), 2))
    print("Calibrated F1:\t", np.round(100*F1(answer_list, all_calibrated_preds, average='macro'), 2))
    print("Majority Baseline F1:\t", np.round(100*F1(answer_list, [max_label]*len(answer_list), average='macro'), 2))
    if args.save_preds:
        results_lookup = [{'file':'regular_predictions.txt', 'list':all_regular_preds}, {'file':'calibrated_predictions.txt', 'list':all_calibrated_preds}, {'file':'ground_truths.txt', 'list':answer_list}]
        for r in results_lookup:
            with open(dest_path + r['file'], 'w+') as f:
                for p in r['list']: f.write(p + '\n')
        task_labels = [l.strip(" ") for l in task_labels]
        for preds in [all_regular_preds, all_calibrated_preds, answer_list]:
            preds = [task_labels.index(l) for l in preds]
