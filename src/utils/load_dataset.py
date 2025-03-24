import torch
from transformers import PreTrainedTokenizer
from src.data.base.datasets import BaseDataset
from src.data.classification import (
    MNLIDataset,
    MRDataset,
    QNLIDataset,
    SST2Dataset,
    TrecDataset,
    YahooDataset
)
from src.data.generation import (
    GSM8KDataset,
    MathDataset,
    SamsumDataset
)
from src.data.qa import MedQADataset, OpenbookQADataset
from src.data.multi_task import BBHDataset, NaturalInstructionsDataset
from CoolPrompt.src.utils.data import BBH_TASKS, NATURAL_INSTRUCTIONS_TASKS


def load_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = 'test',
    prompt: str = None,
    max_seq_length: int = None,
    device: torch.device = None,
    sample: int = None,
    seed: int = 42
) -> BaseDataset:
    dataset_name = dataset_name.lower()
    if dataset_name in BBH_TASKS:
        multi_ds = BBHDataset(
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
            sample=sample,
            seed=seed
        )
        return multi_ds.task(dataset_name)
    if dataset_name in NATURAL_INSTRUCTIONS_TASKS:
        multi_ds = NaturalInstructionsDataset(
            tokenizer=tokenizer,
            split=split,
            prompt=prompt,
            max_seq_length=max_seq_length,
            device=device,
            sample=sample,
            seed=seed
        )
        return multi_ds.task(dataset_name)

    args = {}
    match dataset_name:
        case "mnli": dataset_cls = MNLIDataset
        case "mr": dataset_cls = MRDataset
        case "qnli": dataset_cls = QNLIDataset
        case "sst-2": dataset_cls = SST2Dataset
        case "trec": dataset_cls = TrecDataset
        case "yahoo": dataset_cls = YahooDataset
        case "gsm8k": dataset_cls = GSM8KDataset
        case "math": dataset_cls = MathDataset
        case "samsum": dataset_cls = SamsumDataset
        case "medqa": dataset_cls = MedQADataset
        case "medqa_4_options":
            dataset_cls = MedQADataset
            args['four_options'] = True
        case "openbookqa": dataset_cls = OpenbookQADataset
        case _: raise ValueError("Unsupported dataset name")

    return dataset_cls(
        tokenizer=tokenizer,
        split=split,
        prompt=prompt,
        max_seq_length=max_seq_length,
        device=device,
        sample=sample,
        seed=seed,
        **args
    )
