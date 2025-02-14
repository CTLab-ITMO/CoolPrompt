from torch.utils.data import Dataset 
import transformers
import torch
import pandas as pd 
from typing import Tuple

class SST2Dataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, max_seq_length: int = 50) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path

        self.df = self._read_data()
        self.max_seq_length = max_seq_length

        self.input_ids, self.labels = self._process_data()

    def _read_data(self) -> pd.DataFrame:
        with open(self.data_path, 'r') as file:
            return pd.read_parquet(file)

    def _process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sentences = self.df['input']
        labels = self.df['target']

        messages = [
            {"role": "user", "content": sentence}
            
            for sentence in sentences
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        return input_ids.type(torch.long), torch.Tensor(labels).type(torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[ind], self.labels[ind]
