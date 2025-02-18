from .base_datasets import BaseDataset
from transformers import PreTrainedTokenizer
from typing import Tuple
import torch


class YahooDataset(BaseDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_titles = self.df['question_title']
        questions = self.df['question']
        answers = self.df['answer']
        labels = self.df['target']

        messages = [self.prompt
                    + question_title
                    + "\n"
                    + ("" if question is None else question)
                    + "\nAnswer\n"
                    + answer
                    + "\nResponse:\n"
                    for question_title, question, answer
                    in zip(question_titles, questions, answers)]

        input_ids, attention_mask = self._tokenize(messages)

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )
