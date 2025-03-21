from transformers import PreTrainedTokenizer
import torch
from src.data.base.datasets import BaseClassificationDataset


class MNLIDataset(BaseClassificationDataset):
    """Classification dataset class for MNLI dataset

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
        tokenizer: PreTrainedTokenizer,
        split: str = 'test',
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
        sample: int = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name='mnli',
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
            sample=sample,
            seed=seed
        )
