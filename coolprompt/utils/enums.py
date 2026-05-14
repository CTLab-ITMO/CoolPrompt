from enum import Enum


class Task(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"

    def __str__(self):
        return self.value


class PD_Method(Enum):
    BASE = "base"
    DATASET_BASED = "dataset-based"
