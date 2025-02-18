from typing import List, Sequence


def labels_to_numbers(
    original_labels: List[str],
    ordered_labels: Sequence[str]
) -> List[int]:
    label_projection = {label: ind for ind, label in enumerate(ordered_labels)}
    return [label_projection[label] for label in original_labels]
