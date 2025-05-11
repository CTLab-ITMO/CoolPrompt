"""
Candidate Framework

Provides a data class for a prompt and its training score.
Provides a list wrapper for managing candidates.
"""

from dataclasses import dataclass

@dataclass
class Candidate:
    """
    Represents a candidate with a prompt and its associated training score.

    Attributes:
        prompt (str): The text of the prompt.
        train_score (float): The training score associated with the prompt.
    """
    prompt: str
    train_score: float


class CandidateHistory:
    """
    A class to manage a history of Candidate objects.

    This class provides methods to add, extend, clear, and retrieve candidates
    based on their training scores.

    Attributes:
        candidates (list[Candidate]): A list of Candidate objects.
    """

    def __init__(self, candidates: list[Candidate] | None = None):
        """
        Initializes the CandidateHistory with an optional list of candidates.

        Args:
            candidates (list[Candidate] | None): An optional list of Candidate objects to initialize the history.
        """
        self.candidates = []
        if candidates:
            self.candidates.extend(candidates)

    def add(self, cand: Candidate) -> None:
        """
        Adds a single Candidate to the history.

        Args:
            cand (Candidate): The Candidate object to add.
        """
        self.candidates.append(cand)

    def extend(self, candidates: list[Candidate]) -> None:
        """
        Extends the history with a list of Candidate objects.

        Args:
            candidates (list[Candidate]): A list of Candidate objects to add.
        """
        self.candidates.extend(candidates)

    def clear(self) -> None:
        """
        Clears all candidates from the history.
        """
        self.candidates = []

    def get_highest_scorer(self) -> Candidate:
        """
        Returns the candidate with the highest training score.

        Returns:
            Candidate: The Candidate object with the maximum training score.

        Raises:
            Exception: If there are no candidates in the history.
        """
        if not self.candidates:
            raise Exception("No candidates in history")
        return max(self.candidates, key=lambda cand: cand.train_score)
