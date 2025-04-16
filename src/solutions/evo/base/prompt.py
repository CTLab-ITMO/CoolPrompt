from enum import Enum
from typing import Type


class PromptOrigin(Enum):
    manual = 1
    ape = 2
    evoluted = 3


class Prompt:
    def __init__(
        self,
        text: str,
        origin: PromptOrigin = PromptOrigin.evoluted,
        score: float = None
    ) -> None:
        self.text = text
        if self.text[-1] == '\n':
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
            origin=data['origin'],
            score=data.get('score', None),
        )
