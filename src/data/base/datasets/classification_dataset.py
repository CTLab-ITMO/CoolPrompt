from typing import Tuple, List
from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import BaseDataset
from src.utils.data_utils import labels_to_numbers


class BaseClassificationDataset(BaseDataset):
    """BaseDataset for Classification tasks.

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
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
            sample=sample,
            seed=seed
        )

    def _get_labels(self) -> List[str]:
        """Creates a list of all labels of the dataset.

        Returns:
            List[str]: list of all labels.
        """
        return self._get_data_from_config('labels.json')

    def _get_prompt_template(self) -> str:
        """Extracts prompt template for the dataset from config file.

        Returns:
            str: prompt template.
        """
        return self._get_data_from_config(
            "prompt_templates.json",
            key='classification'
        )

    def _use_prompt_template(self) -> None:
        """Combining prompt template and prompt
        The resulting model input will be stored in self.prompt.
        """
        template = self._get_prompt_template()
        template = template.replace("<PROMPT>", self.prompt)
        self.prompt = template.replace(
            "<LABELS>",
            ', '.join(
                [f"{label}"
                 for ind, label in enumerate(self.labels)])
            )

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processing self.df data (extracting, tokenizing, etc.).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - token input ids
                - attention masks
                - numeric label identificators
        """
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
