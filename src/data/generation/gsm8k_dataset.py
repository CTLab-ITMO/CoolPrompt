from src.data.base.datasets import BaseGenerationDataset
from transformers import PreTrainedTokenizer
import torch


class GSM8KDataset(BaseGenerationDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(
            name='gsm8k',
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
