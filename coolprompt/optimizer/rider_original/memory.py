"""Long-term memory of success and failure patterns (Fig. 1, right)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .population import Candidate

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'-]+")
# Common instruction verbs used to summarize what a prompt asks the model to do.
_ACTION_VERBS = {
    "analyze", "answer", "apply", "avoid", "calculate", "check", "choose",
    "classify", "compare", "compute", "consider", "convert", "count", "define",
    "describe", "determine", "do", "double-check", "ensure", "explain",
    "extract", "focus", "follow", "format", "generate", "give", "identify",
    "include", "infer", "keep", "label", "list", "locate", "match", "note",
    "output", "pick", "provide", "read", "reason", "respond", "return",
    "review", "select", "solve", "state", "summarize", "use", "verify", "write",
}


def extract_pattern(text: str, max_chars: int = 220) -> str:
    """Semantic extraction: the first sentence plus the action verbs used."""

    flat = " ".join(text.split())
    first = _SENTENCE.split(flat, maxsplit=1)[0][:max_chars]
    verbs: list[str] = []
    for word in _WORD.findall(flat):
        lower = word.lower()
        if lower in _ACTION_VERBS and lower not in verbs:
            verbs.append(lower)
    verbs = verbs[:8]
    return f"{first} [actions: {', '.join(verbs)}]" if verbs else first


@dataclass
class LongTermMemory:
    """Top-k success and bottom-k failure patterns, refreshed periodically."""

    patterns: int = 3
    successes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    updates: int = 0

    def update(self, population: list[Candidate], pool: list[Candidate]) -> None:
        """Successes from the kept population, failures from everything evaluated."""

        kept = sorted(
            (c for c in population if c.evaluated), key=lambda c: c.fitness, reverse=True
        )
        if not kept:
            return
        top = kept[: self.patterns]
        tried = sorted((c for c in pool if c.evaluated), key=lambda c: c.fitness)
        bottom = [c for c in tried if c not in top][: self.patterns]
        self.successes = [
            f"{extract_pattern(c.text)} (fitness {c.fitness:.3f})" for c in top
        ]
        self.failures = [
            f"{extract_pattern(c.text)} (fitness {c.fitness:.3f})" for c in bottom
        ]
        self.updates += 1

    def render(self) -> str:
        if not self.successes and not self.failures:
            return ""
        lines = ["Patterns from earlier generations:"]
        if self.successes:
            lines.append("Worked well:")
            lines += [f"- {p}" for p in self.successes]
        if self.failures:
            lines.append("Worked poorly (avoid):")
            lines += [f"- {p}" for p in self.failures]
        return "\n".join(lines)
