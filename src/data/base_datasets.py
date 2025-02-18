from .utils import labels_to_numbers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import pandas as pd
from typing import Tuple, List, Sequence
import os


class BaseDataset(Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.prompt = prompt

        self.df = self._read_data()
        self.max_seq_length = max_seq_length

        self.input_ids, self.attention_mask, self.labels = self._process_data()

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

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        labels = self.df['target']

        messages = [self.prompt + sentence + "\nResponse:\n"
                    for sentence in sentences]

        input_ids, attention_mask = self._tokenize(messages)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.labels[ind]
        )


class BaseNonLabeledDataset(BaseDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        targets = list(self.df['target'])

        messages = [self.prompt + sentence + "\nResponse:\n"
                    for sentence in sentences]

        input_ids, attention_mask = self._tokenize(messages)
        target_ids, _ = self._tokenize(targets)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(target_ids).type(torch.long)
        )


class BaseMultiTaskDataset:

    def __init__(
        self,
        dir_path: str,
        tokenizer: PreTrainedTokenizer,
        prompt: str = "",
        max_seq_lenght: int = None,
        name_split_character: str = "-"
    ) -> None:
        self.dir_path = dir_path
        self.tokenizer = tokenizer
        self.tasks_paths = self._get_tasks()
        self.prompt = prompt
        self.max_seq_length = max_seq_lenght
        self.split_char = name_split_character

    def _get_tasks(self):
        files = os.listdir(self.dir_path)
        task_names = [file.split(self.split_char)[0] for file in files]
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
        return BaseNonLabeledDataset(
            tokenizer=self.tokenizer,
            data_path=task_path,
            prompt=self.prompt,
            max_seq_length=self.max_seq_length
        )


class BaseQADataset(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None,
        labels_set: Sequence[str] = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)
        self.labels_set = labels_set

    def _make_options(self, options: dict) -> str:
        options_list = "\n".join(f"{k}: {v}" for k, v in options.items())
        return "\nOptions:\n" + options_list + "\n"

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diagnoses = self.df['input']
        options = self.df['options']
        answers = list(self.df['target'])
        labels = list(self.df['label'])
        if self.labels_set is None:
            self.labels_set = set(labels)
        labels = labels_to_numbers(labels, self.labels_set)

        messages = [self.prompt
                    + diagnosis
                    + self._make_options(cur_options)
                    + "Response:\n"
                    for diagnosis, cur_options in zip(diagnoses, options)]

        input_ids, attention_mask = self._tokenize(messages)
        self.answers_ids, _ = self._tokenize(answers)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(labels).type(torch.long)
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
