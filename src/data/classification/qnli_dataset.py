from typing import Tuple
import torch
from transformers import PreTrainedTokenizer
from src.data.base.datasets import BaseClassificationDataset
from src.utils.data_utils import labels_to_numbers


class QNLIDataset(BaseClassificationDataset):
    """Classification dataset class for QNLI dataset

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
        seed: int = 42
    ) -> None:
        super().__init__(
            name='qnli',
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
            sample=sample,
            seed=seed
        )

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        questions = self.df['question']
        sentences = self.df['sentence']
        labels = list(self.df['target'])
        numeric_labels = labels_to_numbers(labels, self.labels)

        inputs = [self.prompt.replace(
                    '<INPUT>',
                    f"question : {question}\nsentence : {sentence}\n"
                  )
                  for question, sentence in zip(questions, sentences)]

        input_ids, attention_mask = self._tokenize(inputs)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(numeric_labels).type(torch.long)
        )
