from typing import Tuple, List
from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import BaseDataset
from src.utils.data_utils import labels_to_numbers


class BaseClassificationDataset(BaseDataset):
    """BaseDataset for Classification tasks."""

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
