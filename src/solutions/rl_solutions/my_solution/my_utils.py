import torch
import torch.nn.functional as F
import heapq
import string
from collections import Counter


class TopAccuracyTextsNoDuplicates:
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text, ep):
        if text in self.only_text:
            return False
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (accuracy, len(text), text, ep))
            self.text_map[text] = (len(self.heap) - 1, ep)
        elif accuracy > self.heap[0][0]:
            removed_text = heapq.heappop(self.heap)[2]
            if removed_text in self.text_map:
                self.text_map.pop(removed_text)
            heapq.heappush(self.heap, (accuracy, len(text), text, ep))
            self.text_map[text] = (len(self.heap) - 1, ep)
            self.only_text.append(text)
            return True

    def get_top_texts(self):
        return sorted([(accuracy, text, ep) for accuracy, _, text, ep in self.heap], reverse=True)


def evaluation_classification(
    prompts,
    dataset,
    model,
    tokenizer,
    device,
    verbalizer,
    evaluator,
    id_to_label,
    side='First',
    batch_size=4,
):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, drop_last=False)
    model.eval()
    accuracys = []
    sds = []
    for prompt in prompts:
        total = 0
        correct = 0
        sd = 0
        with torch.no_grad():
            for input_ids, _, label_ids in dataloader:
                
                inputs, targets = decode_input_output(
                    input_ids, label_ids, tokenizer,
                    dataset, evaluator, "classification", id_to_label
                )
                
                softmax_diff, acc = evaluation_soft(
                    [prompt],
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    side=side,
                )
                batch_size = len(targets)
                correct += acc[0] * batch_size
                total += batch_size
                sd += softmax_diff[0]
        accuracy = correct / total
        soft_diff = sd / total
        accuracys.append(torch.Tensor([accuracy]))
        sds.append(torch.Tensor([soft_diff]))

    return accuracys, sds


def evaluation_soft(prompts,
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    Fail_coefficient=1,
                    Success_coefficient=1,
                    side='First',
                    ):
    def _format_prompts(prompts, inputs, side):
        if side == 'First':
            template = "{prompt} Input : {sentence_1} Output:"
        else:
            template = "{sentence_1} {prompt}"
        return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]

    def _get_next_token_index(input_ids):
        return input_ids.shape[1] - 1

    def _get_logits(texts, tokenizer, model, device):
        batch_size = len(texts)
        encoded_inputs = tokenizer(texts, padding='longest', truncation=True,
                                   return_tensors="pt", add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        next_token_indices = _get_next_token_index(encoded_inputs['input_ids'])
        out_logits = token_logits[range(batch_size), next_token_indices, :]
        return out_logits

    accuracies = []
    rewards = []
    model.eval()
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
    batch_size = targets.size(0)
    for prompt in prompts:
        # Get logits
        current_prompts = [prompt for _ in range(batch_size)]
        formatted_templates = _format_prompts(current_prompts, inputs, side=side)
        all_logits = _get_logits(formatted_templates, tokenizer, model, device)

        # Get verbalizer logits
        verbalizer_logits = all_logits[:, verbalizer_ids]
        log_probs = F.softmax(verbalizer_logits, dim=1)

        # Get accuracy
        preds = torch.argmax(log_probs, dim=1).cpu()
        correct_predictions = torch.sum(preds == targets)
        accuracy = correct_predictions.item() / batch_size
        accuracies.append(accuracy)

        # Get reward
        reward = get_reward(all_logits, targets, Fail_coefficient=Fail_coefficient, Success_coefficient=Success_coefficient)
        mean_reward = reward.mean().cpu()
        rewards.append(mean_reward)

    return rewards, accuracies


def get_reward(
    logits,
    labels,
    Fail_coefficient=1,
    Success_coefficient=1,
):
    with torch.no_grad():
        labels = labels.to('cpu')
        logits = logits.to('cpu')
        correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        mask = torch.ones_like(logits)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        masked_logits = logits * mask

        max_other_logits = masked_logits.max(dim=1)[0]
        differences = correct_logits - max_other_logits

    reward = torch.where(differences > 0, differences * Success_coefficient, differences * Fail_coefficient)
    return reward


def evaluation_generation(
    prompts,
    dataset,
    model,
    tokenizer,
    device,
    evaluator
):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False)
    reward = []
    with torch.no_grad():
        for prompt in prompts:
            loss = 0
            total = 0
            for input_ids, _, label_ids in dataloader:
                inputs, labels = decode_input_output(
                    input_ids, label_ids, tokenizer,
                    dataset, evaluator, "generation", None
                )
                template = prompt + inputs[0] + '\n Output : '
                prompt_encoded = tokenizer(template, return_tensors='pt').to(device)
                outputs = model.generate(**prompt_encoded, max_new_tokens=10, do_sample=True)
                r_outputs = tokenizer.decode(
                    outputs[0][len(prompt_encoded[0]):], skip_special_tokens=True).replace('\n', '')
                if isinstance(labels, list):
                    f1 = get_f1_score(r_outputs, labels[0])
                else:
                    f1 = get_f1_score(r_outputs, labels)
                loss += f1
                total += 1
            reward.append(loss/total)
    return reward


def decode_input_output(input_ids, label_ids, tokenizer,
                        dataset, evaluator, task, id_to_label):
    inputs = tokenizer.decode(input_ids, skip_special_tokens=True)
    outputs = evaluator._prepare_labels(tokenizer, dataset, label_ids)
    if task == "classification":
        outputs = [id_to_label[out] for out in outputs]
    return inputs, outputs


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))
