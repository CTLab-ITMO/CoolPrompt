from sklearn.model_selection import train_test_split
from typing import Iterable, List, Optional, Tuple
import numpy as np

from coolprompt.utils.enums import Task


def get_dataset_split(
    dataset: Iterable[str],
    target: Iterable[str],
    validation_size: float,
    train_as_test: bool,
    random_state: Optional[int] = None,
) -> Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
    """Provides a train/val dataset split.

    Args:
        dataset (Iterable[str]):
            Provided dataset.
        target (Iterable[str]):
            Provided targets for the dataset.
        validation_size (float):
            Provided size of validation subset.
        train_as_test (bool):
            Either to use all data for train and validation or split it.
        random_state (Optional[int]):
            Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
            a tuple of train dataset, validation dataset,
            train targets and validation targets.
    """
    if train_as_test:
        return (dataset, dataset, target, target)
    train_data, val_data, train_targets, val_targets = train_test_split(
        dataset, target, test_size=validation_size, random_state=random_state
    )
    return (train_data, val_data, train_targets, val_targets)


def get_stratified_dataset_split(
    dataset: List[str],
    target: List,
    validation_size: float,
    task: Task,
    generation_bins: int = 3,
    random_state: int = 42,
) -> Tuple[List, List, List, List]:
    """Train/val split with stratification by class label (classification)
    or input-length quantile bins (generation).

    Falls back to a plain random split if stratification is not feasible
    (e.g. too few samples per stratum).

    Args:
        dataset: input texts.
        target: ground-truth labels or references.
        validation_size: fraction of data for validation.
        task: Task.CLASSIFICATION or Task.GENERATION.
        generation_bins: number of length quantile bins for generation tasks.
        random_state: random seed for reproducibility.

    Returns:
        (train_data, val_data, train_targets, val_targets)
    """
    dataset = list(dataset)
    target = list(target)

    if task == Task.CLASSIFICATION:
        stratify_labels = [str(t) for t in target]
    else:
        lengths = np.array([len(s) for s in dataset], dtype=np.float64)
        n_bins = max(int(generation_bins), 2)
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(lengths, quantiles))
        stratify_labels = [str(int(np.searchsorted(edges, l, side="right")))
                           for l in lengths]

    try:
        train_data, val_data, train_targets, val_targets = train_test_split(
            dataset, target,
            test_size=validation_size,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError:
        train_data, val_data, train_targets, val_targets = train_test_split(
            dataset, target,
            test_size=validation_size,
            random_state=random_state,
        )

    return train_data, val_data, train_targets, val_targets
