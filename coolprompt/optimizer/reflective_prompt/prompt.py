from enum import Enum
from typing import Type


class PromptOrigin(Enum):
    """Enum type for different prompt origins.
    Prompt origin doesn't affect anything during evolution.
    It is used for more descriptive logs.
    """

    MANUAL = "manual"
    APE = "ape"
    EVOLUTED = "evoluted"
    MUTATED = "mutated"

    @classmethod
    def from_string(cls: Type['PromptOrigin'], string: str) -> 'PromptOrigin':
        """Creates PromptOrigin variable from string description.

        Args:
            string (str): string representation of prompt origin.

        Returns:
            PromptOrigin: enum PromptOrigin variable.
        """
        return cls(string.lower())


class Prompt:
    def __init__(
        self,
        text: str,
        origin: PromptOrigin = PromptOrigin.EVOLUTED,
        score: float = None
    ) -> None:

        """Prompt class.

        Attributes:
            text (str): prompt text.
            origin (PromptOrigin, optional): prompt origin.
                Defaults to PromptOrigin.EVOLUTED.
            score (float, optional): prompt evaluation score. Defaults to None.
        """
        self.text = text
        self.origin = origin
        self.score = score

    def set_score(self, new_score: float) -> None:
        """Records new prompt evaluation score.

        Args:
            new_score (float): new prompt score to set.
        """
        self.score = float(new_score)

    def to_dict(self) -> dict:
        """Creates dictionary representation of prompt.

        Returns:
            dict: created dictionary.
        """
        result = {
            'text': self.text,
            'origin': self.origin.name
        }
        if self.score is not None:
            result['score'] = self.score
        return result

    @classmethod
    def from_dict(
        cls: Type['Prompt'],
        data: dict,
        origin: PromptOrigin = None
    ) -> 'Prompt':
        """Creates Prompt variable from dictionary data.

        Args:
            data (dict): dictionary representation of prompt.
            origin (PromptOrigin, optional):
                can be used to override prompt origin that is stored in data.
                Defaults to None.

        Returns:
            Prompt: created prompt variable.
        """
        if origin:
            data.update(origin=origin.name)
        return cls(
            text=data['text'],
            origin=PromptOrigin.from_string(data['origin']),
            score=data.get('score', None),
        )

    def __str__(self) -> str:
        """Creates string representation of prompt.
        Right now it is just prompt text and evaluation score.

        Returns:
            str: string representation of prompt.
        """
        return f"{self.text}\t{self.score}"
