import json
from typing import Tuple
from transformers import PreTrainedTokenizer
import torch
from src.utils.data_utils import labels_to_numbers
from src.data.base.datasets import BaseClassificationDataset


class BaseQADataset(BaseClassificationDataset):
    """Classification dataset for QA data.

    Attributes:
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
        name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )

    def _make_options(self, options: str) -> str:
        """Creates string that represents list of all possible answers

        Args:
            options (str): string representation of dictionary that maps
                every option ('A', 'B', etc.) to its answer

        Returns:
            str: string representation of the list of all options
        """
        options = json.loads(options)
        options_list = "\n".join(f"{k}: {v}" for k, v in options.items())
        return "\n\nOPTIONS:\n" + options_list + "\n"

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processing self.df data (extracting, tokenizing, etc.).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - token input ids
                - attention masks
                - numeric label identificators
        """
        questions = self.df['input']
        options = self.df['options']
        labels = list(self.df['label'])
        num_labels = labels_to_numbers(labels, self.labels)

        inputs = ["\n\nQUESTION:\n"
                  + question
                  + self._make_options(cur_options)
                  for question, cur_options in zip(questions, options)]

        messages = [self.prompt.replace('<INPUT>', input)
                    for input in inputs]

        input_ids, attention_mask = self._tokenize(messages)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(num_labels).type(torch.long)
        )
