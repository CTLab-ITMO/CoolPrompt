from abc import ABC, abstractmethod
import json
import os
from typing import Tuple, List, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import pandas as pd
from src.utils.data import ALL_DATA_PATH


class BaseDataset(Dataset, ABC):
    """Base abstract class for our datasets.

    Attributes:
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        split: 'test' or 'train' data split. By default is 'test'.
        data_path: a path to file with data.
        config_path: a path to directory with config files
            (such as prompt_templates.json, basic_prompts.json etc.).
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
        df: pandas.DataFrame that contains the data.
        input_ids: torch.Tensor of input token ids for model.
        attention_mask: torch.Tensor of attention masks for model.
        num_labels: torch.Tensor of numeric identificators of the labels.
        sample: number of elements to sample from data
        seed: seed to use while sampling
    """

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
        sample: int = None,
        seed: int = 42
    ) -> None:
        super().__init__()

        self.name = name
        self.tokenizer = tokenizer

        self.split = split
        assert self.split in ['test', 'train']

        self.data_path = self._get_data_path()
        self.config_path = ALL_DATA_PATH
        self.device = device

        self.labels = self._get_labels()

        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = self._get_basic_prompt()

        self._use_prompt_template()

        self.df = self._read_data()

        self.sample = sample
        self.seed = seed

        self._sample_data()

        self.max_seq_length = max_seq_length

        (
            self.input_ids,
            self.attention_mask,
            self.num_labels,
        ) = self._process_data()

        self.input_ids = self.input_ids.to(self.device)
        self.attention_mask = self.attention_mask.to(self.device)
        self.num_labels = self.num_labels.to(self.device)

    def _sample_data(self) -> None:
        """Sampling data from DataFrame."""

        if self.sample is not None:
            self.sample = min(self.sample, len(self.df))

            self.df = self.df.sample(self.sample, random_state=self.seed)

    def _get_data_path(self) -> str:
        """Generates path to data file

        Returns:
            str: path to data
        """
        return os.path.join(
            ALL_DATA_PATH,
            self.name,
            f"{self.split}-00000-of-00001.parquet"
        )

    def _get_labels(self) -> List[str]:
        """Creates a list of all labels of the dataset.

        Returns:
            List[str]: list of all labels.
        """
        return list()

    def get_labels_mapping(self) -> dict:
        """Creates dictionary, that allows mapping
        from string labels to their numeric identificators.

        Returns:
            dict[str, int]: dictionary for mapping.
        """
        return {label: ind for ind, label in enumerate(self.labels)}

    def _get_basic_prompt(self) -> str:
        """Extracts basic prompt for the dataset from config file.

        Returns:
            str: basic prompt.
        """
        return self._get_data_from_config("basic_prompts.json")

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """Extracts prompt template for the dataset from config file.

        Returns:
            str: prompt template.
        """
        pass

    def _get_data_from_config(
        self,
        config_filename: str,
        key: str = None
    ) -> Any:
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
        key = key if key is not None else self.name
        return data.get(key, "")

    @abstractmethod
    def _use_prompt_template(self) -> None:
        """Combining prompt template and prompt
        The resulting model input will be stored in self.prompt.
        """
        pass

    def _read_data(self) -> pd.DataFrame:
        """Reading data from file located at self.data_path.
        By default, all data files have parquet format.

        Returns:
            pd.DataFrame: DataFrame with data.
        """
        with open(self.data_path, 'rb') as file:
            df = pd.read_parquet(file)
        return df

    def _tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizing giving list of texts.

        Args:
            texts (List[str]): list of texts to tokenize.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - token input ids
                - attention masks
        """
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
        """Processing self.df data (extracting, tokenizing, etc.).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - token input ids
                - attention masks
                - numeric label identificators
        """
        pass

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: dataset length.
        """
        return len(self.num_labels)

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the element from dataset on the given position.

        Args:
            ind (int): the position of the element.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                the tuple of element's representation in dataset
                (input ids, attention mask and numeric label).
        """
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.num_labels[ind]
        )
