from sklearn.model_selection import train_test_split
from typing import Iterable, Tuple


def get_dataset_split(
    self,
    dataset: Iterable[str],
    target: Iterable[str],
    validation_size: float,
    train_as_test: bool,
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

    Returns:
        Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
            a tuple of train dataset, validation dataset,
            train targets and validation targets.
    """
    if train_as_test:
        return (dataset, dataset, target, target)
    train_data, val_data, train_targets, val_targets = train_test_split(
        dataset, target, test_size=validation_size
    )
    return (train_data, val_data, train_targets, val_targets)
