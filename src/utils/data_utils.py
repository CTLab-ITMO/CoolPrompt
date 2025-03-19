import os
from typing import List, Sequence


ALL_DATA_PATH = os.path.expanduser('~/autoprompting_data')

INNER_GENERATION_TASKS = set([
    'dyck_languages',
    'multistep_arithmetic_two',
    'object_counting',
    'word_sorting'
])


def labels_to_numbers(
    original_labels: List[str],
    ordered_labels: Sequence[str]
) -> List[int]:
    label_projection = {label: ind for ind, label in enumerate(ordered_labels)}
    return [label_projection[label] for label in original_labels]
