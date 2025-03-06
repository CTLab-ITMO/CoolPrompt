from src.data.base.datasets import BaseQADataset
from transformers import PreTrainedTokenizer
import torch


class MedQADataset(BaseQADataset):
    """Q/A dataset class for MedQA dataset.
    This dataset contains two versions of data:
    4-options data and multiple-options data.

    Attributes:
        name: a string name of the dataset.
        tokenizer: a tokenizer provided for text tokenization.
        data_path: a path to file with data.
        config_path: a path to directory with config files
            (such as prompt_templates.json, basic_prompts.json etc.).
        prompt: a string that describes task for LLM.
        max_seq_length: an integer limit of token sequence.
        device: device where to store tokenized data.
        labels: array of all labels in dataset.
        df: pandas.DataFrame that contains the data.
        input_ids: torch.Tensor of input token ids for model.
        attention_mask: torch.Tensor of attention masks for model.
        num_labels: torch.Tensor of numeric identificators of the labels.
        four_options: boolean value
            (if all the questions will have exact 4 answer options or not)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        config_path: str,
        prompt: str = None,
        max_seq_length: int = None,
        four_options: bool = False,
        device: torch.device = None,
    ) -> None:
        name = "medqa_4_options" if four_options else "medqa"
        self.four_options = four_options

        super().__init__(
            name=name,
            tokenizer=tokenizer,
            data_path=data_path,
            prompt_config_dir_path=config_path,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device
        )
