from enum import Enum


class Method(Enum):
    HYPE = "hype"
    REFLECTIVE = "reflective"
    DISTILL = "distill"

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
