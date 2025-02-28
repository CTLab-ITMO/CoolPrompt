import json
import os
import random
import math
import pdb
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def lowercase_list(lst):
    return [l.lower() for l in lst]

def one_token(label):
    return tokenizer.decode(tokenizer.encode(label, return_tensors='pt')[0][0])

def encode_instruction(task, instruction_structure=['Definition','Prompt','Things to Avoid','Emphasis & Caution', 'Negative Examples Full Explanations', 'Positive Examples Full Explanations'], number_of_examples=0, number_of_instances= 100, null_word=None, data_seed=0, modified={}, args=None):
    random.seed(0) # Ensure the same test set
    with open(args.data_dir+task) as json_file:
        data = json.load(json_file)
    labels = list(set([data["Instances"][i]["output"][0] for i in range(len(data["Instances"])) ]))
    labels.sort()

    assert len(labels) < 25, "Check {} is a classification task.".format(task)
    instances_per_label = number_of_instances // len(labels)
    remainder = number_of_instances % len(labels)
    instance_pools = {label:{'indices':[]} for label in labels} 
    for i, inst in enumerate(data["Instances"]):
        label = inst['output'][0]
        instance_pools[label]['indices'].append(i)
    remaining = 0
    test_pools = {}
    
    for l, label in enumerate(labels):
        if len(instance_pools[label]['indices']) >= 4 + instances_per_label: #leave out some examples for Definition + Examples (hard-coded)
            num = instances_per_label
            if l < remainder: num += 1
            
            test_pools[label] = random.sample(instance_pools[label]['indices'], num)
            instance_pools[label]['indices'] = [i for i in instance_pools[label]['indices'] if i not in test_pools[label]]
        else: 
            num = len(instance_pools[label]['indices']) - 4 
            remaining += instances_per_label - num
            
            test_pools[label] = random.sample(instance_pools[label]['indices'], num)
            instance_pools[label]['indices'] = [i for i in instance_pools[label]['indices'] if i not in test_pools[label]]
            
    all_remaining_indices = []
    remaining = number_of_instances - sum([len(t) for t in test_pools.values()])
    for label in labels: 
        all_remaining_indices.extend(instance_pools[label]['indices'])
    remaining_test = random.sample(all_remaining_indices, remaining)
    
    for t in remaining_test: 
        label = data['Instances'][t]['output'][0]
        test_pools[label].append(t)
        instance_pools[label]['indices'].remove(t)

    indexlist = []
    for label in labels: 
        indexlist.extend(test_pools[label])
    assert len(indexlist) == number_of_instances, pdb.set_trace()

    random.seed(data_seed)
    if number_of_examples == -1: 
        total_num_examples = 1
    else: 
        total_num_examples = number_of_examples * len(labels)
    pos_examples = {label:[] for label in labels}
    for eg in data["Positive Examples"]: 
        label = eg['output']
        try: pos_examples[label].append(eg)
        except: pdb.set_trace()
    for label in labels:
        for id in instance_pools[label]['indices']:
            inst = data["Instances"][id]
            inst['output'] = inst['output'][0]
            pos_examples[label].append(inst)
    
    chosen_examples = []
    if number_of_examples > 0 : 
        for label in labels: 
            chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    elif number_of_examples == -1: 
        label = random.sample(labels, 1)
        chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    assert len(chosen_examples) == total_num_examples
    random.shuffle(chosen_examples)

    generic_instruction=''
    for i in instruction_structure:
        if i!='Positive Examples Full Only' and i!='Positive Examples Full Explanations' and i!='Negative Examples Full Explanations':
            if data[i]!='-':
                if i in modified.keys():
                    data[i] = modified[i]
                data[i] = data[i].replace('\n' + 'Things to avoid: -', '')
                data[i] = data[i].replace('\n' + 'Emphasis & Caution: -', '')
                if generic_instruction=='':
                    generic_instruction=generic_instruction+i+': '+data[i].strip() 
                else:
                    generic_instruction=generic_instruction+"\n"+i+': '+data[i].strip() 
        elif i=='Positive Examples Full Only' :
            for j in range(total_num_examples):
                if 'examples' in modified.keys():
                    if generic_instruction!='':  
                        generic_instruction=generic_instruction+"\n"+'input: '+modified['examples'][j]['input'] + "\n"+ 'output: '+ one_token(modified['examples'][j]['output'])
                    else:
                        generic_instruction=generic_instruction+'input: '+modified['examples']['input'] + "\n"+ 'output: '+ one_token(modified['examples'][j]['output'])
                else:
                    if generic_instruction!='':  
                        generic_instruction=generic_instruction+"\n"+'input: '+chosen_examples[j]['input'] + "\n"+ 'output: '+ one_token(chosen_examples[j]['output'])
                    else:
                        generic_instruction=generic_instruction+'input: '+chosen_examples[j]['input'] + "\n"+ 'output: '+ one_token(chosen_examples[j]['output'])
                
        elif i=='Positive Examples Full Explanations' : #This mode of Natural Instructions not supported
            assert False
            
        elif i=='Negative Examples Full Explanations' : #This mode of Natural Instructions not supported
            assert False
            
    promptlist=[]
    answerlist=[]

    for i in range(number_of_instances):
        if null_word is None:
            if 'input' in modified.keys():
                if generic_instruction!= '': 
                    prompt=generic_instruction+"\n"+'input: '+data['Instances'][indexlist[i]]['input']+" " + modified['input'] + "\n"+"output:"
                else: 
                    prompt='input: '+data['Instances'][indexlist[i]]['input']+"\n"+"output:"
            else:   
                if generic_instruction!= '': 
                    prompt=generic_instruction+"\n"+'input: '+data['Instances'][indexlist[i]]['input']+"\n"+"output:"
                else: 
                    prompt='input: '+data['Instances'][indexlist[i]]['input']+"\n"+"output:"
        else:
            if generic_instruction!='': 
                prompt=generic_instruction+"\n"+'input: '+null_word+"\n"+"output:"
            else: 
                prompt='input: '+null_word+"\n"+"output:"
        if 'Completion' in labels[0]:
            prompt = prompt + ' Completion'
        promptlist.append(prompt)
        answer = data['Instances'][indexlist[i]]['output'][0].strip(".").replace('Completion ', '')
        answer = one_token(answer)
        answerlist.append(answer)
    
    return promptlist, answerlist, indexlist

def training_encode_instruction(task, instruction_structure =['Definition','Prompt','Things to Avoid','Emphasis & Caution', 'Negative Examples Full Explanations', 'Positive Examples Full Explanations'], number_of_examples=0, number_of_instances= 100, null_word=None, data_seed=0, modified={}, args=None):
    random.seed(0) # Ensure the same test set
    with open(args.data_dir+task) as json_file:
        data = json.load(json_file)
    labels = list(set([data["Instances"][i]["output"][0] for i in range(len(data["Instances"])) ]))
    labels.sort()
    assert len(labels) < 25, "Check {} is a classification task.".format(task)
    instances_per_label = number_of_instances // len(labels)
    remainder = number_of_instances % len(labels)
    instance_pools = {label:{'indices':[]} for label in labels} 
    for i, inst in enumerate(data["Instances"]):
        label = inst['output'][0]
        instance_pools[label]['indices'].append(i)
    remaining = 0
    test_pools = {}
    
    for l, label in enumerate(labels):
        if len(instance_pools[label]['indices']) >= 4 + instances_per_label: #see comment in function above
            num = instances_per_label
            if l < remainder: num += 1

            test_pools[label] = random.sample(instance_pools[label]['indices'], num)
            instance_pools[label]['indices'] = [i for i in instance_pools[label]['indices'] if i not in test_pools[label]]
        else: 
            num = len(instance_pools[label]['indices']) - 4
            remaining += instances_per_label - num
            
            test_pools[label] = random.sample(instance_pools[label]['indices'], num)
            instance_pools[label]['indices'] = [i for i in instance_pools[label]['indices'] if i not in test_pools[label]]
            
    all_remaining_indices = []
    remaining = number_of_instances - sum([len(t) for t in test_pools.values()])
    for label in labels: 
        all_remaining_indices.extend(instance_pools[label]['indices'])
    remaining_test = random.sample(all_remaining_indices, remaining)
    
    for t in remaining_test: 
        label = data['Instances'][t]['output'][0]
        test_pools[label].append(t)
        instance_pools[label]['indices'].remove(t)

    indexlist = []
    for label in labels: 
        indexlist.extend(test_pools[label])
    assert len(indexlist) == number_of_instances, pdb.set_trace()

    random.seed(data_seed)
    if number_of_examples == -1: 
        total_num_examples = 1
    else: 
        total_num_examples = number_of_examples * len(labels)
    pos_examples = {label:[] for label in labels}
    for eg in data["Positive Examples"]:
        label = eg['output']
        pos_examples[label].append(eg)
    for label in labels:
        for id in instance_pools[label]['indices']:
            inst = data["Instances"][id]
            inst['output'] = inst['output'][0]
            pos_examples[label].append(inst)
    
    chosen_examples = []
    if number_of_examples > 0 : 
        for label in labels: chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    elif number_of_examples == -1: 
        label = random.sample(labels, 1)
        chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    assert len(chosen_examples) == total_num_examples
    random.shuffle(chosen_examples)

    train_indexlist = list(range(len(data['Instances'])))
    train_indexlist = [i for i in train_indexlist if i not in indexlist and data['Instances'][i] not in chosen_examples]

    dev_len = round(0.1*len(train_indexlist))
    dev_indexlist = random.sample(train_indexlist, dev_len)
    train_indexlist = [i for i in train_indexlist if i not in dev_indexlist]

    generic_instruction=''
    for i in instruction_structure:
        if i!='Positive Examples Full Only' and i!='Positive Examples Full Explanations' and i!='Negative Examples Full Explanations':
            if data[i]!='-':
                if i in modified.keys():
                    data[i] = modified[i]
                data[i] = data[i].replace('\n' + 'Things to avoid: -', '')
                data[i] = data[i].replace('\n' + 'Emphasis & Caution: -', '')
                # pdb.set_trace()
                if generic_instruction=='':
                    generic_instruction=generic_instruction+i+': '+data[i].strip() 
                else:
                    generic_instruction=generic_instruction+"\n"+i+': '+data[i].strip() 
        elif i=='Positive Examples Full Only' :
            for j in range(total_num_examples):
                if generic_instruction!='':  
                    generic_instruction=generic_instruction+"\n"+'input: '+chosen_examples[j]['input'] + "\n"+ 'output: '+ one_token(chosen_examples[j]['output'])
                else:
                    generic_instruction=generic_instruction+'input: '+chosen_examples[j]['input'] + "\n"+ 'output: '+one_token(chosen_examples[j]['output'])
                    
        elif i=='Positive Examples Full Explanations' : #This mode of Natural Instructions not supported
            assert False
            
        elif i=='Negative Examples Full Explanations' : #This mode of Natural Instructions not supported
            assert False
            
    promptlist=[]
    answerlist=[]

    for i in range(number_of_instances):
        if null_word is None:
            if generic_instruction!= '': prompt=generic_instruction+"\n"+'input: '+data['Instances'][indexlist[i]]['input']+"\n"+"output:"
            else: prompt='input: '+data['Instances'][indexlist[i]]['input']+"\n"+"output:"
        else:
            if generic_instruction!='': prompt=generic_instruction+"\n"+'input: '+null_word+"\n"+"output:"
            else: prompt='input: '+null_word+"\n"+"output:"
        if 'Completion' in labels[0]:
            prompt = prompt + ' Completion'
        promptlist.append(prompt)
        answer = data['Instances'][indexlist[i]]['output'][0].strip(".").replace('Completion ', '')
        answer = one_token(answer)
        answerlist.append(answer)

    train_promptlist=[]
    train_answerlist=[]

    for i in range(len(train_indexlist)):
        if null_word is None:
            if generic_instruction!= '': prompt=generic_instruction+"\n"+'input: '+data['Instances'][train_indexlist[i]]['input']+"\n"+"output:"
            else: prompt='input: '+data['Instances'][train_indexlist[i]]['input']+"\n"+"output:"
        else:
            if generic_instruction!='': prompt=generic_instruction+"\n"+'input: '+null_word+"\n"+"output:"
            else: prompt='input: '+null_word+"\n"+"output:"
        if 'Completion' in labels[0]:
            prompt = prompt + ' Completion'
        train_promptlist.append(prompt)
        train_answer = data['Instances'][train_indexlist[i]]['output'].strip(".").replace('Completion ', '')
        train_answer = one_token(train_answer)
        train_answerlist.append(train_answer)

    dev_promptlist=[]
    dev_answerlist=[]

    for i in range(len(dev_indexlist)):
        if null_word is None:
            if generic_instruction!= '': prompt=generic_instruction+"\n"+'input: '+data['Instances'][dev_indexlist[i]]['input']+"\n"+"output:"
            else: prompt='input: '+data['Instances'][dev_indexlist[i]]['input']+"\n"+"output:"
        else:
            if generic_instruction!='': prompt=generic_instruction+"\n"+'input: '+null_word+"\n"+"output:"
            else: prompt='input: '+null_word+"\n"+"output:"
        if 'Completion' in labels[0]:
            prompt = prompt + ' Completion'
        dev_promptlist.append(prompt)
        dev_answer = data['Instances'][dev_indexlist[i]]['output'].strip(".").replace('Completion ', '')
        dev_answer = one_token(dev_answer)
        dev_answerlist.append(dev_answer)
    return promptlist, answerlist, indexlist, train_promptlist, train_answerlist, train_indexlist, dev_promptlist, dev_answerlist, dev_indexlist
