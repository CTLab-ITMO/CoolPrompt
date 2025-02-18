from .base_datasets import BaseNonLabeledDataset
from transformers import PreTrainedTokenizer


class SamsumDataset(BaseNonLabeledDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = "",
        max_seq_length: int = None
    ) -> None:
        super().__init__(tokenizer, data_path, prompt, max_seq_length)
