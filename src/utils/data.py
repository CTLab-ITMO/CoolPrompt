import os
from typing import List, Sequence


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
