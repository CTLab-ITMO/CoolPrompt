from .utils import labels_to_numbers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import pandas as pd
from typing import Tuple, List
import os
import json
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt_config_dir_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        super().__init__()

        self.name = name
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.config_path = prompt_config_dir_path

        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = self._get_basic_prompt()

        self._use_prompt_template()

        self.df = self._read_data()
        self.max_seq_length = max_seq_length

        (
            self.input_ids,
            self.attention_mask,
            self.num_labels,
        ) = self._process_data()

    def _get_basic_prompt(self) -> str:
        return self._get_data_from_config("basic_prompts.json")

    @abstractmethod
    def _get_prompt_template(self) -> str:
        pass

    def _get_data_from_config(self, config_filename: str, key: str = None):
        path = os.path.join(self.config_path, config_filename)
        with open(path, 'r') as f:
            data = json.load(f)
        key = key if key is not None else self.name
        return data.get(key, "")

    @abstractmethod
    def _use_prompt_template(self) -> None:
        pass

    def _read_data(self) -> pd.DataFrame:
        with open(self.data_path, 'rb') as file:
            df = pd.read_parquet(file)
        return df

    def _tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return (input_ids, attention_mask)

    @abstractmethod
    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def __len__(self):
        return len(self.num_labels)

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.num_labels[ind]
        )


class BaseClassificationDataset(BaseDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt_config_dir_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        self.labels = self._get_labels()

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=prompt_config_dir_path,
            prompt=prompt,
            max_seq_length=max_seq_length
        )

    def _get_labels(self) -> List[str]:
        return self._get_data_from_config('labels.json')

    def _get_prompt_template(self) -> str:
        return self._get_data_from_config(
            "prompt_templates.json",
            key='classification'
        )

    def _use_prompt_template(self) -> None:
        template = self._get_prompt_template()
        template = template.replace("<PROMPT>", self.prompt)
        self.prompt = template.replace("<LABELS>", ','.join(self.labels))

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        labels = list(self.df['target'])
        numeric_labels = labels_to_numbers(labels, self.labels)

        inputs = [self.prompt.replace('<INPUT>', sentence)
                  for sentence in sentences]

        input_ids, attention_mask = self._tokenize(inputs)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(numeric_labels).type(torch.long)
        )


class BaseGenerationDataset(BaseDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt_config_dir_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        self.response_prefix = self._get_response_prefix()

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=prompt_config_dir_path,
            prompt=prompt,
            max_seq_length=max_seq_length
        )

    def _get_response_prefix(self) -> str:
        # disable this functionality right now,
        # as it ruins all the process without data file
        # I think the use of response prefixes should be discussed at first
        # return self._get_data_from_config('response_prefixes.json')
        return ""

    def _get_prompt_template(self) -> str:
        return self._get_data_from_config(
            "prompt_templates.json",
            key='generation'
        )

    def _use_prompt_template(self) -> None:
        template = self._get_prompt_template()
        template = template.replace("<PROMPT>", self.prompt)
        self.prompt = template.replace(
            "<RESPONSE_PREFIX>",
            self.response_prefix
        )

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        target_outputs = list(self.df['target'])

        inputs = [self.prompt.replace('<INPUT>', sentence)
                  for sentence in sentences]

        input_ids, attention_mask = self._tokenize(inputs)

        target_ids, _ = self._tokenize(target_outputs)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            target_ids.type(torch.long)
        )


INNER_GENERATION_TASKS = set([
    'dyck_languages',
    'multistep_arithmetic_two',
    'object_counting',
    'word_sorting'
])


class InnerTaskClassificationDataset(BaseClassificationDataset):

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None,
    ) -> None:
        self.base_name = base_name
        super().__init__(name, tokenizer, data_path, prompt, max_seq_length)

    def _get_data_from_config(self, config_filename: str, key: str = None):
        path = os.path.join(self.config_path, config_filename)
        with open(path, 'r') as f:
            data = json.load(f)

        if key is not None:
            return data[key]
        return data[self.base_name][self.name]


class InnerTaskGenerationDataset(BaseGenerationDataset):

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None,
    ) -> None:
        self.base_name = base_name
        super().__init__(name, tokenizer, data_path, prompt, max_seq_length)

    def _get_data_from_config(self, config_filename: str, key: str = None):
        path = os.path.join(self.config_path, config_filename)
        with open(path, 'r') as f:
            data = json.load(f)

        if key is not None:
            return data[key]
        return data[self.base_name][self.name]


class BaseMultiTaskDataset:

    def __init__(
        self,
        name: str,
        dir_path: str,
        tokenizer: PreTrainedTokenizer,
        prompt: str = None,
        max_seq_lenght: int = None
    ) -> None:
        self.name = name
        self.dir_path = dir_path
        self.tokenizer = tokenizer
        self.tasks_paths = self._get_tasks()
        self.prompt = prompt
        self.max_seq_length = max_seq_lenght

    def _get_tasks(self):
        files = os.listdir(self.dir_path)
        task_names = [file.split("-")[0] for file in files]
        return {
            task_name: path_to_task
            for task_name, path_to_task in zip(task_names, files)
        }

    def task(self, task_name: str) -> Dataset:
        task_path = self.tasks_paths.get(task_name, None)
        if task_path is None:
            raise ValueError(
                f"""Bad task name
                All supported tasks: [{','.join(self.tasks_paths.keys())}]"""
            )

        if task_name in INNER_GENERATION_TASKS:
            return InnerTaskGenerationDataset(
                base_name=self.name,
                name=task_name,
                tokenizer=self.tokenizer,
                data_path=task_path,
                prompt=self.prompt,
                max_seq_length=self.max_seq_length
            )

        return InnerTaskClassificationDataset(
            base_name=self.name,
            name=task_name,
            tokenizer=self.tokenizer,
            data_path=task_path,
            prompt=self.prompt,
            max_seq_length=self.max_seq_length
        )


class BaseQADataset(BaseClassificationDataset):
    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt_config_dir_path: str,
        prompt: str = None,
        max_seq_length: int = None,
    ) -> None:
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=prompt_config_dir_path,
            prompt=prompt,
            max_seq_length=max_seq_length
        )

    def _make_options(self, options: dict) -> str:
        options_list = "\n".join(f"{k}: {v}" for k, v in options.items())
        return "\n\nOPTIONS:\n" + options_list + "\n"

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        questions = self.df['input']
        options = self.df['options']
        answers = list(self.df['target'])
        labels = list(self.df['label'])
        num_labels = labels_to_numbers(labels, self.labels)

        inputs = ["\n\nQUESTION:\n"
                  + question
                  + self._make_options(cur_options)
                  for question, cur_options in zip(questions, options)]

        messages = [self.prompt.replace('<INPUT>', input)
                    for input in inputs]

        input_ids, attention_mask = self._tokenize(messages)
        self.answers_ids, _ = self._tokenize(answers)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(num_labels).type(torch.long)
        )

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.labels[ind],
            self.answers_ids[ind],
        )
