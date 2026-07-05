from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DataSpec:
    task_description: Optional[str] = field(
        default=None,
        metadata={"hint": "One sentence: what should the model do?"},
    )
    domain: Optional[str] = field(
        default=None,
        metadata={"hint": "Subject-matter area, e.g. 'medical QA', 'social-media sentiment'."},
    )
    input_description: Optional[str] = field(
        default=None,
        metadata={"hint": "What does one input look like? Mention format, length, language."},
    )
    output_description: Optional[str] = field(
        default=None,
        metadata={"hint": "What should the output look like? Format, allowed values, no explanation?"},
    )
    label_set: Optional[list[str]] = field(
        default=None,
        metadata={"hint": "Classification only. All valid output labels."},
    )
    constraints: Optional[list[str]] = field(
        default=None,
        metadata={"hint": "Hard rules every example must follow."},
    )
    corner_cases: Optional[list[str]] = field(
        default=None,
        metadata={"hint": "Tricky situations to cover, e.g. 'sarcastic reviews', 'very short inputs'."},
    )
    language: Optional[str] = field(
        default=None,
        metadata={"hint": "Primary language. Defaults to English."},
    )
    additional_notes: Optional[str] = field(
        default=None,
        metadata={"hint": "Extra style or topic guidance for the generator."},
    )

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def to_prompt_block(self) -> str:
        pairs = {
            "Task description": self.task_description,
            "Domain": self.domain,
            "Input format": self.input_description,
            "Output format": self.output_description,
            "Valid labels": ", ".join(self.label_set) if self.label_set else None,
            "Constraints": "; ".join(self.constraints) if self.constraints else None,
            "Corner cases": "; ".join(self.corner_cases) if self.corner_cases else None,
            "Language": self.language,
            "Additional notes": self.additional_notes,
        }
        lines = [f"  {k}: {v}" for k, v in pairs.items() if v is not None]
        return "[User Specification]\n" + "\n".join(lines) if lines else ""
