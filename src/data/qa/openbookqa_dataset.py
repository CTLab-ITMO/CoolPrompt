from src.data.base.datasets import BaseQADataset
from transformers import PreTrainedTokenizer
import torch


class OpenbookQADataset(BaseQADataset):

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
            name="openbookqa",
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
