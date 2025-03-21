import os
from typing import List, Sequence
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


ALL_DATA_PATH = os.path.expanduser('~/autoprompting_data')

INNER_GENERATION_TASKS = set([
    'dyck_languages',
    'multistep_arithmetic_two',
    'object_counting',
    'word_sorting'
])

BBH_TASKS = set([
    'boolean_expressions',
    'hyperbaton',
    'temporal_sequences',
    'object_counting',
    'disambiguation_qa',
    'logical_deduction_three_objects',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'causal_judgement',
    'date_understanding',
    'ruin_names',
    'word_sorting',
    'geometric_shapes',
    'movie_recommendation',
    'salient_translation_error_detection',
    'formal_fallacies',
    'penguins_in_a_table',
    'dyck_languages',
    'multistep_arithmetic_two',
    'navigate',
    'reasoning_about_colored_objects',
    'tracking_shuffled_objects_three_objects',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'sports_understanding',
    'snarks',
    'web_of_lies'
])

NATURAL_INSTRUCTIONS_TASKS = set([
    'task021',
    'task050',
    'task069'
])


def labels_to_numbers(
    original_labels: List[str],
    ordered_labels: Sequence[str]
) -> List[int]:
    label_projection = {label: ind for ind, label in enumerate(ordered_labels)}
    return [label_projection[label] for label in original_labels]


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
