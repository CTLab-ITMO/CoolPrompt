from torch.utils.data import Dataset
import transformers
import torch
import pandas as pd
from typing import Tuple


class SST2Dataset(Dataset):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.prompt = prompt

        self.df = self._read_data()
        self.max_seq_length = max_seq_length

        self.input_ids, self.attention_mask, self.labels = self._process_data()

    def _read_data(self) -> pd.DataFrame:
        with open(self.data_path, 'rb') as file:
            df = pd.read_parquet(file)
        return df

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        labels = self.df['target']

        messages = [self.prompt + sentence + "\n\tResponse:\n"
                    for sentence in sentences]

        inputs = self.tokenizer(
            messages,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return (
            input_ids.type(torch.long),
            attention_mask.type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[ind],
            self.attention_mask[ind],
            self.labels[ind]
        )
