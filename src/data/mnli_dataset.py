from .base_datasets import BaseDataset
from .utils import labels_to_numbers
from transformers import PreTrainedTokenizer
from typing import Tuple
import torch


class MNLIDataset(BaseDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences1 = self.df['sentence1']
        sentences2 = self.df['sentence2']
        labels = list(self.df['gold_label'])

        messages = [self.prompt
                    + "\nSentence one:"
                    + sentence1
                    + "\nSentence two:\n"
                    + sentence2
                    + "\nResponse:\n"
                    for sentence1, sentence2 in zip(sentences1, sentences2)]

        input_ids, attention_mask = self._tokenize(messages)
        numeric_labels = labels_to_numbers(labels, set(labels))

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(numeric_labels).type(torch.long)
        )
