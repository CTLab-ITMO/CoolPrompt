from enum import Enum
from typing import Type, List, Dict


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
    def from_string(cls: Type["PromptOrigin"], string: str) -> "PromptOrigin":
        """Creates PromptOrigin variable from string description.

        Args:
            string (str): string representation of prompt origin.

        Returns:
            PromptOrigin: enum PromptOrigin variable.
        """
        return cls(string.lower())


class BadExample:
    """Bad Example class

    Attributes:
        input (str): input of the example.
        output (str): model output for the example.
        correct (str): correct output of the example.
    """

    def __init__(self, input: str, output: str, correct: str):
        self.input = input
        self.output = output
        self.correct = correct

    def to_dict(self) -> dict:
        """Creates dictionary representation of bad example.

        Returns:
            dict: created dictionary.
        """

        return {
            "input": self.input,
            "output": self.output,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls: Type["BadExample"], data: dict) -> "BadExample":
        """Creates BadExample variable from dictionary data.

        Args:
            data (dict): dictionary representation of bad example.

        Returns:
            BadExample: created bad example variable.
        """

        return cls(
            input=data["input"], output=data["output"], correct=data["correct"]
        )


class Prompt:
    """Prompt candidate with origin, score, and optional failed examples."""

    def __init__(
        self,
        text: str,
        origin: PromptOrigin = PromptOrigin.EVOLUTED,
        score: float = None,
        bad_examples: List[BadExample] = [],
    ) -> None:
        """Prompt class.

        Attributes:
            text (str): prompt text.
            origin (PromptOrigin, optional): prompt origin.
                Defaults to PromptOrigin.EVOLUTED.
            score (float, optional): prompt evaluation score. Defaults to None.
            bad_examples (List[BadExample]): a list of
                bad examples for the prompt.
        """

        self.text = text
        self.origin = origin
        self.score = score
        self.bad_examples = bad_examples

    def set_score(self, new_score: float) -> None:
        """Records new prompt evaluation score.

        Args:
            new_score (float): new prompt score to set.
        """

        self.score = float(new_score)

    def set_bad_examples(self, bad_examples: List[Dict[str, str]]) -> None:
        """Stores provided bad examples."""

        self.bad_examples = [
            BadExample(
                input=example["input"],
                output=example["output"],
                correct=example["correct"],
            )
            for example in bad_examples
        ]

    def to_dict(self) -> dict:
        """Creates dictionary representation of prompt.

        Returns:
            dict: created dictionary.
        """

        result = {
            "text": self.text,
            "origin": self.origin.name,
        }
        if self.score is not None:
            result["score"] = self.score
        if len(self.bad_examples) > 0:
            result["bad_examples"] = [ex.to_dict() for ex in self.bad_examples]
        return result

    @classmethod
    def from_dict(
        cls: Type["Prompt"], data: dict, origin: PromptOrigin = None
    ) -> "Prompt":
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
            text=data["text"],
            origin=PromptOrigin.from_string(data["origin"]),
            score=data.get("score", None),
            bad_examples=[
                BadExample.from_dict(bad_example_data)
                for bad_example_data in data.get("bad_examples", [])
            ],
        )

    def __str__(self) -> str:
        """Creates string representation of prompt.
        Right now it is just prompt text and evaluation score.

        Returns:
            str: string representation of prompt.
        """

        return f"{self.text}\t{self.score}"
