import torch
import numpy as np
import transformers
import src.data as data
import my_utils as utils

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig


class StablePromptAgent:
    def __init__(self,
                 task,
                 log_file,
                 target_model_name,
                 agent_model_name,
                 quantization_config,
                 dataset: data.BaseDataset,
                 evaluator: data.BaseNLPEvaluator,
                 metric,
                 examples_sample=30,
                 reward_calc_sample=40,
                 update_calc_sample=30,
                 meta_prompt='''
            I gave a friend an instruction and five inputs.
            The friend read the instruction and wrote an output for every one of the inputs.
            Here are the input-output pairs: \n
            ''',
                 max_prompt_length=150,
                 prompt_per_example=4,
                 num_example=5,
                 update_term=5,
                 update_threshold=0.05
                 ):
        self.task = task
        self.log_file_name = log_file
        self.dataset = dataset
        self.evaluator = evaluator
        self.metric = metric
        self.prompt_per_example = prompt_per_example
        self.max_prompt_length = max_prompt_length
        self.num_example = num_example
        self.meta_prompt = meta_prompt
        self.update_term = update_term
        self.update_threshold = update_threshold
        self.queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)

        self.device = 'cuda:0'

        # load agent model
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.agent_tokenizer = AutoTokenizer.from_pretrained(agent_model_name)
        self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token
        self.agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            agent_model_name,
            device_map='auto',
            peft_config=self.lora_config,
            quantization_config=quantization_config
        )
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            agent_model_name,
            device_map='auto',
            peft_config=self.lora_config,
            quantization_config=quantization_config
        )

        # load target model
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            device_map='auto',
            quantization_config=quantization_config
        )
        self.target_model.config.pad_token_id = self.target_tokenizer.eos_token_id

        if self.task == "classification":
            labels_mapping = self.dataset(
                tokenizer=self.target_tokenizer, device=self.device
                ).get_labels_mapping()
            self.id_to_label = {v: k for k, v in labels_mapping.items()}

        # create ppo trainer
        ppo_config = PPOConfig(
            model_name=agent_model_name,
            learning_rate=1e-5,
            batch_size=prompt_per_example,
            mini_batch_size=prompt_per_example,
        )
        self.ppo_trainer = PPOTrainer(ppo_config, self.agent_model,
                                      self.ref_model, self.agent_tokenizer)

        # generation kwargs setting
        self.generation_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.agent_tokenizer.eos_token_id,
            "max_new_tokens": max_prompt_length,
            "min_length": -1,
        }

        # set model generate params
        terminators = [
            self.target_tokenizer.eos_token_id,
            self.target_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.model_generate_params = {
            "max_new_tokens": 256,
            "eos_token_id": terminators
        }

        # load learning datasets
        ds = self.dataset(tokenizer=self.target_tokenizer, split="train", 
                          sample=(examples_sample + reward_calc_sample + update_calc_sample),
                          device=self.device)
        self.examples_dataset, self.reward_calc_dataset, self.update_calc_dataset = torch.utils.data.random_split(
            ds, [examples_sample, reward_calc_sample, update_calc_sample])

    def _evaluate_prompt(self, prompt, metric, sample, split="test"):
        metrics = self.evaluator.evaluate(
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            eval_ds=self.dataset(tokenizer=self.target_tokenizer, sample=sample,
                                 split=split, prompt=prompt, device=self.device),
            batch_size=16,
            model_generate_args=self.model_generate_params,
        )
        return metrics[metric]

    def _evaluate_list_of_prompts(self, prompts, metric, sample, split="test"):
        return [self._evaluate_prompt(prompt, metric, sample, split) for prompt in prompts]

    def _get_examples(self):
        dataloader = torch.utils.data.DataLoader(self.examples_dataset, sampler=torch.utils.data.RandomSampler(
            self.examples_dataset, num_samples=self.num_example))
        examples = ""
        for input_ids, _, label_ids in dataloader:
            inp, out = utils.decode_input_output(
                input_ids, label_ids, self.agent_tokenizer,
                self.examples_dataset, self.evaluator,
                self.task, self.id_to_label)
            examples += "Input : " + inp + "\nOutput : " + out
        return examples

    def _get_query(self):
        examples = self._get_examples()
        query_text = [
            {"role": "user", "content": self.meta_prompt + '\n' + examples},
            {"role": "assistant", "content": "The Instruction is : "}
        ]
        return self.agent_tokenizer.apply_chat_template(
            query_text,
            return_tensors='pt'
        ).view(-1).to(self.device)

    def _decode_tensors(self, tensors):
        return [self.agent_tokenizer.decode(tensor.squeeze(), skip_special_tokens=True) for tensor in tensors]

    def _update_referense_model(self, bs, query_encoded, sample):
        response_tensors, ref_response_tensors = self.ppo_trainer.generate(
            query_encoded.view(-1),
            **self.generation_kwargs,
            return_prompt=False,
            num_return_sequences=bs,
            generate_ref_response=True)

        prompts = self._decode_tensors(response_tensors)
        ref_prompts = self._decode_tensors(ref_response_tensors)

        if self.task == "classification":
            verbalizer = list(self.id_to_label.values())
            _, acc = utils.evaluation_classification(
                prompts=prompts,
                dataset=self.update_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                verbalizer=verbalizer,
                evaluator=self.evaluator,
                id_to_label=self.id_to_label
            )
            _, ref_acc = utils.evaluation_classification(
                prompts=ref_prompts,
                dataset=self.update_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                verbalizer=verbalizer,
                evaluator=self.evaluator,
                id_to_label=self.id_to_label
            )
        elif self.task == "generation":
            acc = utils.evaluation_generation(
                prompts=prompts,
                dataset=self.update_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                evaluator=self.evaluator
            )
            ref_acc = utils.evaluation_generation(
                prompts=ref_prompts,
                dataset=self.update_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                evaluator=self.evaluator
            )
            pass

        mean_acc = np.mean(np.array(acc))
        mean_ref_acc = np.mean(np.array(ref_acc))

        diff = mean_acc - mean_ref_acc
        if diff > self.update_threshold:
            self.ppo_trainer.ref_model = self.ppo_trainer.model
            return 1
        elif diff < -self.update_threshold:
            self.ppo_trainer.model = self.ppo_trainer.ref_model
            return -1
        return 0

    def _get_rewards(self, prompts):
        if self.task == "classification":
            softmax_diff, accuracys = utils.evaluation_classification(
                prompts=prompts,
                dataset=self.reward_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                verbalizer=list(self.id_to_label.values()),
                evaluator=self.evaluator,
                id_to_label=self.id_to_label
            )
            return [0.05 * softmax_diff[i] + 3 * accuracys[i] for i in range(len(prompts))]
        elif self.task == 'generation':
            rewards = utils.evaluation_generation(
                prompts=prompts,
                dataset=self.reward_calc_dataset,
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                device=self.device,
                evaluator=self.evaluator
            )
            return rewards
        return []

    def train(self, epochs):
        with open(self.log_file_name, 'w') as log_file:
            change_num = 0

            for ep in tqdm(range(epochs)):
                query_encoded = self._get_query()
                response_tensors = self.ppo_trainer.generate(
                    query_encoded,
                    **self.generation_kwargs,
                    return_prompt=False,
                    num_return_sequences=self.prompt_per_example
                )

                used_prompts = self._decode_tensors(response_tensors)

                if sum([len(p) for p in used_prompts]) < self.prompt_per_example * 10:
                    break

                rewards = self._get_rewards(used_prompts)
                batch_size = len(np.array(rewards))
                rewards = [torch.tensor(reward) for reward in rewards]
                for i in range(len(rewards)):
                    self.queue.add(rewards[i].item(), used_prompts[i], ep)

                self.ppo_trainer.step(
                    [query_encoded.view(-1) for _ in range(batch_size)],
                    [response for response in response_tensors],
                    rewards
                )
                rewards = torch.stack(rewards)

                if ep != 0 and ep % self.update_term == 0:
                    change_num += self._update_referense_model(batch_size, query_encoded)
                    if change_num < 0:
                        change_num = 0

    def test(self, test_sample=100):
        prompt_queue = self.queue.get_top_texts()
        new_acc = self._evaluate_list_of_prompts(
            [prompt[1] for prompt in prompt_queue],
            metric=self.metric,
            sample=test_sample,
            split="test"
        )
        print(len(prompt_queue), new_acc)
        for i in range(len(prompt_queue)):
            print('prompt : ', prompt_queue[i][1], 'acc : ', new_acc[i])
        max_new_acc = np.max(np.array(new_acc))
        with open('results.txt', "a") as f:
            f.write(str(max_new_acc) + '\n')


if __name__ == "__main__":
    model_name = "t-bank-ai/T-lite-instruct-0.1"
    qconf = transformers.BitsAndBytesConfig(load_in_8bit=True)
    sp_agent = StablePromptAgent(
        task="classification",
        log_file="log.txt",
        target_model_name=model_name,
        agent_model_name=model_name,
        quantization_config=qconf,
        dataset=data.SST2Dataset,
        evaluator=data.TextClassificationEvaluator,
        metric="f1",
    )
    sp_agent.train(
        epochs=50,  # original code uses 100 for fewshot, 30 for others
    )
    sp_agent.test(test_sample=100)
