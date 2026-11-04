"""Population members and the error cases that drive error-directed evolution."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_ids = itertools.count(1)


@dataclass
class ErrorCase:
    """A validation example the current best prompt got wrong."""

    input: str
    output: str
    expected: str
    score: float

    def render(self, max_chars: int = 600) -> str:
        def clip(text: str) -> str:
            text = str(text).strip()
            return text if len(text) <= max_chars else text[:max_chars] + "..."

        return (
            f"Input: {clip(self.input)}\n"
            f"Model output: {clip(self.output)}\n"
            f"Expected: {clip(self.expected)}\n"
            f"Score: {self.score:.2f}"
        )


@dataclass
class Candidate:
    """A prompt with its validation fitness and provenance."""

    text: str
    fitness: float = float("-inf")
    generation: int = 0
    operator: str = "init"
    parents: tuple[int, ...] = ()
    errors: list[ErrorCase] = field(default_factory=list)
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def evaluated(self) -> bool:
        return self.fitness != float("-inf")
