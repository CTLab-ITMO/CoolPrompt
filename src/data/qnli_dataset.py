from .base_datasets import BaseClassificationDataset
from .utils import labels_to_numbers
from transformers import PreTrainedTokenizer
from typing import Tuple
import torch


class QNLIDataset(BaseClassificationDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        super().__init__(
            name='qnli',
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length
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
