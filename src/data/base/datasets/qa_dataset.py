from typing import Tuple
from transformers import PreTrainedTokenizer
import torch
from src.utils.data_utils import labels_to_numbers
from src.data.base.datasets import BaseClassificationDataset


class BaseQADataset(BaseClassificationDataset):
    """Classification dataset for QA data."""

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt_config_dir_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=prompt_config_dir_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )

    def _make_options(self, options: dict[str, str]) -> str:
        """Creates string that represents list of all possible answers

        Args:
            options (dict[str, str]): dictionary that maps
                every option ('A', 'B', etc.) to its string representation

        Returns:
            str: string representation of the list of all options
        """
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

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the element from dataset on the given position.

        Args:
            ind (int): the position of the element.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                the tuple of element's representation in dataset
                (input ids, attention mask, numeric label).
        """
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.labels[ind]
        )
