from src.data.base.datasets import BaseMultiTaskDataset
from transformers import PreTrainedTokenizer
import torch


class BBHDataset(BaseMultiTaskDataset):
    """Compilation of Big Bench Hard tasks

    Attributes:
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        data_path: a path to directory with data.
        config_path: a path to directory with config files
            (such as prompt_templates.json, basic_prompts.json etc.).
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(
            name='bbh',
            dir_path=data_path,
            config_path=config_path,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
