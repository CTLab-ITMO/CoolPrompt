from src.data.base.datasets import BaseMultiTaskDataset
from transformers import PreTrainedTokenizer
import torch


class NaturalInstructionsDataset(BaseMultiTaskDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(
            name='natural_instructions',
            dir_path=data_path,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
