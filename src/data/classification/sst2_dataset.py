from src.data.base.datasets import BaseClassificationDataset
from transformers import PreTrainedTokenizer
import torch


class SST2Dataset(BaseClassificationDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None
    ) -> None:
        super().__init__(
            name='sst-2',
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
        )
