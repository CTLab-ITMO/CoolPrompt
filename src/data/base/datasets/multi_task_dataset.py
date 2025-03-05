import json
import os
from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import (
    BaseDataset,
    BaseClassificationDataset,
    BaseGenerationDataset
)
from src.utils.data_utils import INNER_GENERATION_TASKS


class InnerTaskClassificationDataset(BaseClassificationDataset):
    """Classification dataset class that represents a task
    from multi-task dataset like BBH or Natural Instructions.

    Attributes:
        base_name: the global name of multi-task dataset this task is part of.
    """

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.base_name = base_name
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
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
    """

    def __init__(
        self,
        base_name: str,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.base_name = base_name
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
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
        dir_path: a path to directory with all tasks data.
        tokenizer: a tokenizer provided for text tokenization.
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
    """

    def __init__(
        self,
        name: str,
        dir_path: str,
        tokenizer: PreTrainedTokenizer,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        self.name = name
        self.dir_path = dir_path
        self.tokenizer = tokenizer
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
                max_seq_length=self.max_seq_length,
                device=self.device
            )

        return InnerTaskClassificationDataset(
            base_name=self.name,
            name=task_name,
            tokenizer=self.tokenizer,
            data_path=task_path,
            prompt=self.prompt,
            max_seq_length=self.max_seq_length,
            device=self.device
        )
