"""Candidate Framework.

Provides a data class for a prompt and its training score.
Provides a list wrapper for managing candidates.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Candidate:
    """Represents a candidate with a prompt and its associated training score.

    Attributes:
        prompt (str): The text of the prompt.
        train_score (float): The training score associated with the prompt.
    """
    prompt: str
    train_score: float


class CandidateHistory:
    """A class to manage a history of Candidate objects.

    This class provides methods to add, extend, clear, and retrieve candidates
    based on their training scores.

    Attributes:
        candidates: A list of Candidate objects.
    """

    def __init__(self, candidates: Optional[List[Candidate]] = None) -> None:
        """Initializes the history with an optional list of candidates.

        Args:
            candidates (Optional[List[Candidate]], optional): An optional list
                of Candidate objects to initialize the history with. Defaults
                to None.
        """
        self.candidates: List[Candidate] = []
        if candidates:
            self.candidates.extend(candidates)

    def add(self, candidate: Candidate) -> None:
        """Adds a single Candidate to the history.

        Args:
            candidate (Candidate): The Candidate object to add.
        """
        self.candidates.append(candidate)

    def extend(self, candidates: List[Candidate]) -> None:
        """Extends the history with a list of Candidate objects.

        Args:
            candidates (List[Candidate]): A list of Candidate objects to add.
        """
        self.candidates.extend(candidates)

    def clear(self) -> None:
        """Clears all candidates from the history."""
        self.candidates = []

    def get_highest_scorer(self) -> Candidate:
        """Returns the candidate with the highest training score.

        Returns:
            Candidate: Candidate with the best score.

        Raises:
            ValueError: If there are no candidates in the history.
        """
        if not self.candidates:
            raise ValueError("No candidates in history")
        return max(
            self.candidates,
            key=lambda candidate: candidate.train_score
        )
