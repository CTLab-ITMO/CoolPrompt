import torch
from transformers import PreTrainedTokenizer
from src.data.base.datasets import BaseMultiTaskDataset


class BBHDataset(BaseMultiTaskDataset):
    """Compilation of Big Bench Hard tasks

    Attributes:
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        split: 'test' or 'train' data split. By default is 'test'.
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(
            name='bbh',
            split=split,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
