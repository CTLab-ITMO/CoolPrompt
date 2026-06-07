from enum import Enum


class Task(Enum):
    """Supported high-level task families."""

    CLASSIFICATION = "classification"
    GENERATION = "generation"

    def __str__(self):
        return self.value


class PD_Method(Enum):
    """Problem-description generation strategies."""

    BASE = "base"
    DATASET_BASED = "dataset-based"
