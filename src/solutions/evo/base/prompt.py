from enum import Enum
from typing import Type


class PromptOrigin(Enum):
    MANUAL = "manual"
    APE = "ape"
    EVOLUTED = "evoluted"

    @classmethod
    def from_string(cls, string: str):
        return cls(string.lower())


class Prompt:
    def __init__(
        self,
        text: str,
        origin: PromptOrigin = PromptOrigin.EVOLUTED,
        score: float = None
    ) -> None:
        self.text = text
        if self.text and self.text[-1] == '\n':
            self.text = self.text[:-1]
        self.origin = origin
        self.score = score

    def set_score(self, new_score: float) -> None:
        self.score = new_score

    def to_dict(self) -> dict:
        result = {
            'text': self.text,
            'origin': self.origin.name
        }
        if self.score:
            result['score'] = self.score
        return result

    @classmethod
    def from_dict(
        cls: Type['Prompt'],
        data: dict,
        origin: PromptOrigin = None
    ) -> 'Prompt':
        if origin:
            data.update(origin=origin.name)
        return cls(
            text=data['text'],
            origin=PromptOrigin.from_string(data['origin']),
            score=data.get('score', None),
        )

    def __str__(self) -> str:
        return f"{self.text}\t{self.score}"
