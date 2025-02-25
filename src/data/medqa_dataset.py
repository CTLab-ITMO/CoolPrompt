from .base_datasets import BaseQADataset
from transformers import PreTrainedTokenizer


class MedQADataset(BaseQADataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        four_options: bool = False,
    ) -> None:
        name = "medqa_4_options" if four_options else "medqa"

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length
        )
