import json
import os
from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import (
    BaseDataset,
    BaseClassificationDataset,
    BaseGenerationDataset
)
from src.utils.data_utils import INNER_GENERATION_TASKS, ALL_DATA_PATH


class InnerTaskClassificationDataset(BaseClassificationDataset):
    """Classification dataset class that represents a task
    from multi-task dataset like BBH or Natural Instructions.

    Attributes:
        base_name: the global name of multi-task dataset this task is part of.
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        split: 'test' or 'train' data split. By default is 'test'.
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
        df: pandas.DataFrame that contains the data.
        input_ids: torch.Tensor of input token ids for model.
        attention_mask: torch.Tensor of attention masks for model.
        num_labels: torch.Tensor of numeric identificators of the labels.
    """

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.base_name = base_name

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )

    def _get_data_path(self) -> str:
        """Generates path to data file

        Returns:
            str: path to data
        """
        return os.path.join(
            ALL_DATA_PATH,
            self.base_name,
            self.name,
            f"{self.split}-00000-of-00001.parquet"
        )

    def _get_data_from_config(self, config_filename: str, key: str = None):
        """Provides general operation of extracting
        string data from json config file.
        By default self.name will be used as a key.

        Args:
            config_filename (str): the name of config file.
            key (str, optional): the key used to retrieve the target data
                                 from parsed json object. Defaults to None,
                                 by default will be replaced with self.name.

        Returns:
            Any: the extracted object.
        """
        path = os.path.join(self.config_path, config_filename)
        with open(path, 'r') as f:
            data = json.load(f)

        if key is not None:
            return data[key]
        return data[self.base_name][self.name]


class InnerTaskGenerationDataset(BaseGenerationDataset):
    """Generation dataset class that represents a task
    from multi-task dataset like BBH or Natural Instructions.

    Attributes:
        base_name: the global name of multi-task dataset this task is part of.
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        split: 'test' or 'train' data split. By default is 'test'.
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
        df: pandas.DataFrame that contains the data.
        input_ids: torch.Tensor of input token ids for model.
        attention_mask: torch.Tensor of attention masks for model.
        num_labels: torch.Tensor of numeric identificators of the labels.
        response_prefix: a string prefix that can be
            added right before model output generation.
            By default is empty string.
    """

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.base_name = base_name

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )

    def _get_data_path(self) -> str:
        """Generates path to data file

        Returns:
            str: path to data
        """
        return os.path.join(
            ALL_DATA_PATH,
            self.base_name,
            self.name,
            f"{self.split}-00000-of-00001.parquet"
        )

    def _get_data_from_config(self, config_filename: str, key: str = None):
        path = os.path.join(self.config_path, config_filename)
        with open(path, 'r') as f:
            data = json.load(f)

        if key is not None:
            return data[key]
        return data[self.base_name][self.name]


class BaseMultiTaskDataset(object):
    """Class that provides multi-task dataset implementation.

    Attributes:
        name: a string name of the dataset.
        split: 'test' or 'train' data split. By default is 'test'.
        tokenizer: a tokenizer provided for text tokenization.
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
    """

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.split = split
        self.dir_path = os.path.join(
            ALL_DATA_PATH,
            name
        )
        self.tasks_paths = self._get_tasks()
        self.prompt = prompt
        self.max_seq_length = max_seq_length
        self.device = device

    def _get_tasks(self) -> dict:
        """Extracts all paths to tasks parquet files.

        Returns:
            dict[str, str]: mapping for every task name to its filepath.
        """
        files = os.listdir(self.dir_path)
        task_names = [file.split("-")[0] for file in files]
        return {
            task_name: path_to_task
            for task_name, path_to_task in zip(task_names, files)
        }

    def task(self, task_name: str) -> BaseDataset:
        """Creates dataset for given task name

        Args:
            task_name (str): the name of the task.

        Returns:
            torch.utils.data.Dataset:
                dataset that provides access to the task data.

        Raises:
            ValueError: An error occurred creating task
                for non-exstisning task name.
        """
        task_path = self.tasks_paths.get(task_name, None)

        if task_path is None:
            raise ValueError(
                f"""Invalid task name
                All supported tasks: [{','.join(self.tasks_paths.keys())}]"""
            )

        task_path = os.path.join(self.dir_path, task_path)

        if task_name in INNER_GENERATION_TASKS:
            return InnerTaskGenerationDataset(
                base_name=self.name,
                name=task_name,
                tokenizer=self.tokenizer,
                split=self.split,
                prompt=self.prompt,
                max_seq_length=self.max_seq_length,
                device=self.device
            )

        return InnerTaskClassificationDataset(
            base_name=self.name,
            name=task_name,
            tokenizer=self.tokenizer,
            split=self.split,
            prompt=self.prompt,
            max_seq_length=self.max_seq_length,
            device=self.device
        )
