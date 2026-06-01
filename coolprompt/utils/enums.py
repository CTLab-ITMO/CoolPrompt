from enum import Enum


class Method(Enum):
    HYPE = "hype"
    REFLECTIVE = "reflective"
    DISTILL = "distill"
    PE2 = "pe2"
    PE2_SGR = "pe2_sgr"
    APE = "ape"
    OPRO = "opro"

    def is_data_driven(self) -> bool:
        if self is Method.HYPE:
            return False
        return True

    def __str__(self):
        return self.value


class Task(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"

    def __str__(self):
        return self.value


class PD_Method(Enum):
    BASE = "base"
    DATASET_BASED = "dataset-based"
