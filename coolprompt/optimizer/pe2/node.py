"""Lightweight node for PE2 beam search."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """A node in the PE2 beam search tree.

    Args:
        timestamp (int): The training step at which this node was created.
        id (int): Unique identifier for this node.
        prompt (str): The prompt text held by this node.
        parent (Optional[int]): ID of the parent node, or None for root.
        scores (dict): Mapping of split name to evaluation score.
        n_child (int): Number of child nodes spawned from this node.
    """

    timestamp: int
    id: int
    prompt: str
    parent: Optional[int] = None
    scores: dict = field(default_factory=dict)
    n_child: int = 0

    def register_score(self, score: float, split_name: str) -> None:
        """Records an evaluation score for a given split.

        Args:
            score (float): The metric score.
            split_name (str): Name of the data split (e.g. "val").
        """
        self.scores[split_name] = score
