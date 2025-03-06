from typing import Tuple
from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import BaseDataset


class BaseGenerationDataset(BaseDataset):
    """BaseDataset for Generation tasks.

    Attributes:
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
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
        response_prefix: a string prefix that can be
            added right before model output generation.
            By default is empty string.
    """

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
        self.response_prefix = self._get_response_prefix()

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=prompt_config_dir_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )

    def _get_response_prefix(self) -> str:
        """Returns response prefix for the dataset

        Returns:
            str: response prefix
        """

        # disabled functionality for right now,
        # as it ruins all the process without data file
        # I think the use of response prefixes should be discussed at first
        # return self._get_data_from_config('response_prefixes.json')
        return ""

    def _get_prompt_template(self) -> str:
        """Extracts prompt template for the dataset from config file.

        Returns:
            str: prompt template.
        """
        return self._get_data_from_config(
            "prompt_templates.json",
            key='generation'
        )

    def _use_prompt_template(self) -> None:
        """Combining prompt template and prompt
        The resulting model input will be stored in self.prompt.
        """
        template = self._get_prompt_template()
        template = template.replace("<PROMPT>", self.prompt)
        self.prompt = template.replace(
            "<RESPONSE_PREFIX>",
            self.response_prefix
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
