from src.data.base.datasets import BaseClassificationDataset
from transformers import PreTrainedTokenizer


class MNLIDataset(BaseClassificationDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        super().__init__(
            name='mnli',
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length
        )
