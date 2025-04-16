import torch
import numpy as np
import wandb
import utils
import transformers
import src.data as data

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig


class StablePromptAgent:
    def __init__(self,
                 target_model_name,
                 agent_model_name,
                 quantization_config,
                 dataset: data.BaseDataset,
                 evaluator: data.BaseNLPEvaluator,
                 max_prompt_length,
                 epochs,
                 meta_prompt,
                 prompt_per_example=4,
                 num_example=5,
                 update_term=5,
                 update_threshold=0.05
                 ):
        self.evaluator = evaluator
        self.prompt_per_example = prompt_per_example
        self.max_prompt_length = max_prompt_length
        self.num_example = num_example
        self.epochs = epochs
        self.meta_prompt = meta_prompt
        self.update_term = update_term
        self.update_threshold = update_threshold

        self.device = 'cuda:0'
        wandb.init()

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

        # load dataset
        self.train_dataset = dataset(tokenizer=self.agent_tokenizer) # todo load
        self.validation_dataset = dataset(tokenizer=self.agent_tokenizer)
        self.test_dataset = dataset(tokenizer=self.agent_tokenizer)
        print('train dataset size : ', len(train_dataset))
        print('test dataset size : ', len(test_dataset))

        # create ppo trainer
        ppo_config = PPOConfig(
            model_name=agent_model_name,
            learning_rate=1e-5,
            batch_size=prompt_per_example,
            mini_batch_size=prompt_per_example,
            log_with='wandb',
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

        terminators = [
            self.target_tokenizer.eos_token_id,
            self.target_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.model_generate_params = {
            "max_new_tokens": 256,
            "eos_token_id": terminators
        }

    def _evaluate_prompt(self, prompt):
        metrics = self.evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=self.data(tokenizer=self.tokenizer, sample=100, split=split, prompt=prompt, device=self.device),
            batch_size=args.batch_size,
            model_generate_args=self.model_generate_args,
        )
        return metrics if split == "test" else metrics["f1"]

    def _get_examples(self):
        pass

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

    def _update_referense_model(self, bs, query_encoded, change_num):
        response_tensors, ref_response_tensors = self.ppo_trainer.generate(
            query_encoded.view(-1),
            **self.generation_kwargs,
            return_prompt=False,
            num_return_sequences=bs,
            generate_ref_response=True)
        used_prompts = self._decode_tensors(response_tensors)
        ref_used_prompts = self._decode_tensors(ref_response_tensors)

        acc = self.evaluator.evaluate(
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            eval_ds=self.dataset,
            batch_size=32,
            model_generate_args=self.model_generate_params
        )
        ref_acc = utils.evaluation(
            ref_used_prompts,
            self.validation_dataset,
            self.target_model,
            self.target_tokenizer,
            self.device,
            self.verbalizer.values(),
        )
                
        print('acc : ', acc)
        print('ref_acc : ', ref_acc)
        mean_acc = np.mean(np.array(acc))
        mean_ref_acc = np.mean(np.array(ref_acc))
        diff = mean_acc - mean_ref_acc
        if diff > self.update_threshold:
            self.ppo_trainer.ref_model = self.ppo_trainer.model
            print('update ref model')
            change_num += 1
        elif diff < -self.update_threshold:
            self.ppo_trainer.model = self.ppo_trainer.ref_model
            print('rollback model')
            change_num -= 1
        if change_num < 0:
            change_num = 0

        wandb.log({
            'change_num': change_num,
            'valid_acc': mean_acc,
            'ref_valid_acc': mean_ref_acc,
        })

        return change_num

    def train(self):
        queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
        change_num = 0

        for ep in tqdm(range(self.epochs)):
            max_total_loss = 0
            min_total_loss = 0
            mean_total_loss = 0
            sum_total_loss = 0

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

            rewards = []
            if self.metrics == 'multiple_choice_grade':
                accuracys, softmax_diff = utils.evaluation_sd(
                    used_prompts,
                    self.validation_dataset,
                    self.target_model,
                    self.target_tokenizer,
                    self.device,
                    self.verbalizer.values(),
                    soft_diff=True,
                )
                rewards = [0.05 * softmax_diff[i] + 3 * accuracys[i] for i in range(len(used_prompts))]
            elif self.metrics == 'exact_str_match':
                rewards, accuracys = utils.evaluation_generation(
                    used_prompts,
                    self.validation_dataset,
                    self.target_model,
                    self.target_tokenizer,
                    self.device,
                )

            np_rewards = np.array(rewards)
            rewards = [torch.tensor(reward) for reward in rewards]
            for i in range(len(rewards)):
                print('reward : ', rewards[i].item(), 'acc :', accuracys[i], ' prompt : ', used_prompts[i], '\n')
                queue.add(rewards[i].item(), used_prompts[i], ep)

            bs = len(np_rewards)
            _ = self.ppo_trainer.step(
                [query_encoded.view(-1) for i in range(bs)],
                [response for response in response_tensors],
                rewards)
            rewards = torch.stack(rewards)
            mean_reward = torch.mean(rewards)
            max_reward = torch.max(rewards)
            mean_total_loss += mean_reward
            max_total_loss += max_reward
            min_total_loss += torch.min(rewards)
            sum_total_loss += torch.sum(rewards)
            
            wandb.log({
                'rewards': rewards,
                'mean_reward': mean_reward,
                'max_reward': max_reward,
            })

            if ep != 0 and ep % self.update_term == 0:
                change_num = self._update_referense_model(bs, query_encoded, change_num)

            wandb.log({
                'mean_loss': mean_total_loss,
                'max_loss': max_total_loss,
                'min_loss': min_total_loss,
                'sum_loss': sum_total_loss,
            })

        print('Final test Start')
        prompt_queue = queue.get_top_texts()
        new_acc = utils.evaluation(
            [prompt[1] for prompt in prompt_queue],
            self.test_dataset,
            self.target_model,
            self.target_tokenizer,
            self.device,
            self.verbalizer.values(),
        )
        print(len(prompt_queue), new_acc)
        for i in range(len(prompt_queue)):
            print('prompt : ', prompt_queue[i][1], 'acc : ', new_acc[i])
        max_new_acc = np.max(np.array(new_acc))
        wandb.log({
            'final_acc': max_new_acc,
            'final_mean_acc': np.mean(np.array(new_acc))
        })
        with open('results.txt', "a") as f:
            f.write(str(max_new_acc) + '\n')


if __name__ == "__main__":
    model_name = "t-bank-ai/T-lite-instruct-0.1"
    qconf = transformers.BitsAndBytesConfig(load_in_8bit=True)
    sp_agent = StablePromptAgent(
        target_model_name=model_name,
        agent_model_name=model_name,
        quantization_config=qconf,
        dataset=data.SST2Dataset,
        evaluator=data.TextClassificationEvaluator,
        max_prompt_length=150,
        epochs=50,  # original code uses 100 for fewshot, 30 for others
        meta_prompt='''
            I gave a friend an instruction and five inputs. 
            The friend read the instruction and wrote an output for every one of the inputs.
            Here are the input-output pairs: \n
            ''',
    )
    sp_agent.train()
