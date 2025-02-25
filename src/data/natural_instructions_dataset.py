from .base_datasets import BaseMultiTaskDataset
from transformers import PreTrainedTokenizer


class NaturalInstructionsDataset(BaseMultiTaskDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        prompt: str = None,
        max_seq_length: int = None
    ) -> None:
        super().__init__(
            name='natural_instructions',
            dir_path=data_path,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_lenght=max_seq_length
        )
