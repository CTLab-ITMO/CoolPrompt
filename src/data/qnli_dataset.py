from .base_datasets import BaseDataset
from transformers import PreTrainedTokenizer
from typing import Tuple
import torch


class QNLIDataset(BaseDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        questions = self.df['question']
        sentences = self.df['sentence']
        labels = self.df['label']

        messages = [self.prompt
                    + question
                    + "\nSentence:\n"
                    + sentence
                    + "\nResponse:\n"
                    for question, sentence in zip(questions, sentences)]

        input_ids, attention_mask = self._tokenize(messages)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )
