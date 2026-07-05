from __future__ import annotations

import json
import os
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field

TaskType = Literal[
    "classification",
    "generation",
    "summarization",
    "QA",
    "translation",
    "extraction",
    "evaluation",
    "other",
]

TWEET_EMOTION_CLASSIFICATION = "tweet_emotion_classification"
SCHOOL_MATH_REASONING = "school_math_reasoning"
CONCEPT_TO_SENTENCE_GENERATION = "concept_to_sentence_generation"
CONTEXT_QUESTION_ANSWERING = "context_question_answering"
TEXT_SUMMARIZATION = "text_summarization"

SUPPORTED_TASK_AREAS = (
    TWEET_EMOTION_CLASSIFICATION,
    SCHOOL_MATH_REASONING,
    CONCEPT_TO_SENTENCE_GENERATION,
    CONTEXT_QUESTION_ANSWERING,
    TEXT_SUMMARIZATION,
)

TASK_AREA_TO_DATASET: dict[str, str] = {
    TWEET_EMOTION_CLASSIFICATION: "tweeteval",
    SCHOOL_MATH_REASONING: "gsm8k",
    CONCEPT_TO_SENTENCE_GENERATION: "common_gen",
    CONTEXT_QUESTION_ANSWERING: "squad_v2",
    TEXT_SUMMARIZATION: "xsum",
}

DATASET_LABEL_SETS: dict[str, set[str]] = {
    "tweeteval": {"anger", "joy", "optimism", "sadness"},
}


class IOFormat(BaseModel):
    input_description: str = Field(
        description="Concise description of one input sample (format, length, language, content type)."
    )
    output_description: str = Field(
        description="Concise description of the expected output (format, type, value constraints)."
    )
    input_constraints: list[str] = Field(
        default_factory=list,
        description="Hard input-format constraints: length, language, casing, required structure.",
    )
    output_constraints: list[str] = Field(
        default_factory=list,
        description="Hard output-format constraints: label-only, JSON shape, length, no extra text.",
    )


class CornerCase(BaseModel):
    name: str = Field(description="Short human-readable name for the corner-case pattern.")
    description: str = Field(description="What makes this pattern difficult, ambiguous, or unusual.")
    example_hint: str = Field(description="Brief generation hint to guide the LLM.")


class TaskSpec(BaseModel):
    domain: str = Field(
        description="Subject-matter domain, e.g. 'social-media sentiment', 'legal summarisation'."
    )
    task_type: TaskType = Field(description="High-level task family.")
    task_summary: str = Field(description="One-sentence description of what the model must do.")
    io_format: IOFormat = Field(description="Input and output format details.")
    key_skills: list[str] = Field(description="Atomic capabilities required. Aim for 4–8 items.")
    constraints: list[str] = Field(description="Rules every valid answer must follow. Aim for 3–6 items.")
    typical_errors: list[str] = Field(description="Common model mistakes. Aim for 3–6 items.")
    corner_cases: list[CornerCase] = Field(description="Tricky realistic patterns to cover. Aim for 4–8.")
    language: str = Field(default="English", description="Primary language of inputs and outputs.")
    label_set: list[str] | None = Field(
        default=None,
        description="Exhaustive valid labels for classification; null otherwise.",
    )
    matched_dataset: str | None = Field(
        default=None,
        description="Benchmark dataset slug that best matches this task, or null.",
    )
    additional_notes: str | None = Field(
        default=None,
        description="Extra guidance for generating realistic, diverse examples.",
    )

    def update(
            self,
            *,
            domain: str | None = None,
            task_summary: str | None = None,
            input_description: str | None = None,
            output_description: str | None = None,
            input_constraints: list[str] | None = None,
            output_constraints: list[str] | None = None,
            key_skills: list[str] | None = None,
            constraints: list[str] | None = None,
            typical_errors: list[str] | None = None,
            corner_cases: list[CornerCase] | None = None,
            language: str | None = None,
            label_set: list[str] | None = None,
            matched_dataset: str | None = None,
            additional_notes: str | None = None,
    ) -> "TaskSpec":
        updates = {}

        if domain is not None:
            updates["domain"] = domain
        if task_summary is not None:
            updates["task_summary"] = task_summary
        if key_skills is not None:
            updates["key_skills"] = key_skills
        if constraints is not None:
            updates["constraints"] = constraints
        if typical_errors is not None:
            updates["typical_errors"] = typical_errors
        if corner_cases is not None:
            updates["corner_cases"] = corner_cases
        if language is not None:
            updates["language"] = language
        if label_set is not None:
            updates["label_set"] = label_set
        if matched_dataset is not None:
            updates["matched_dataset"] = matched_dataset
        if additional_notes is not None:
            updates["additional_notes"] = additional_notes

        io_updates = {}

        if input_description is not None:
            io_updates["input_description"] = input_description
        if output_description is not None:
            io_updates["output_description"] = output_description
        if input_constraints is not None:
            io_updates["input_constraints"] = input_constraints
        if output_constraints is not None:
            io_updates["output_constraints"] = output_constraints

        if io_updates:
            updates["io_format"] = self.io_format.model_copy(update=io_updates)

        return self.model_copy(update=updates)

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TaskSpec":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_data_spec_code(self) -> str:

        def _quote(value: str) -> str:
            return repr(value)

        def _list_block(name: str, values: list[str] | None, indent: str = "    ") -> list[str]:
            if not values:
                return []

            lines = [f"{indent}{name}=["]
            lines.extend(f"{indent}    {_quote(value)}," for value in values)
            lines.append(f"{indent}],")
            return lines

        lines = ["DataSpec("]

        lines.append(f"    task_description={_quote(self.task_summary)},")
        lines.append(f"    domain={_quote(self.domain)},")
        lines.append(f"    input_description={_quote(self.io_format.input_description)},")
        lines.append(f"    output_description={_quote(self.io_format.output_description)},")

        if self.label_set:
            lines.extend(_list_block("label_set", self.label_set))

        lines.extend(_list_block("constraints", self.constraints))

        if self.corner_cases:
            corner_cases = [
                f"{case.name}: {case.description}"
                for case in self.corner_cases
            ]
            lines.extend(_list_block("corner_cases", corner_cases))

        if self.language:
            lines.append(f"    language={_quote(self.language)},")

        if self.additional_notes:
            lines.append(f"    additional_notes={_quote(self.additional_notes)},")

        lines.append(")")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self._pretty()

    def __repr__(self) -> str:
        return self._pretty()

    def _pretty(self) -> str:
        def _bullet(items: list) -> str:
            return "\n".join(f"│    • {i}" for i in items) if items else "│    —"

        corner = "\n".join(
            f"│    • {c.name}: {c.description}" for c in self.corner_cases
        ) or "│    —"

        lines = [
            "╭─ TaskSpec " + "─" * 50,
            f"│  domain        {self.domain}",
            f"│  task_type     {self.task_type}",
            f"│  summary       {self.task_summary}",
            "│",
            f"│  input         {self.io_format.input_description}",
            f"│  output        {self.io_format.output_description}",
        ]

        if self.label_set:
            lines += [f"│  labels        {', '.join(self.label_set)}"]

        if self.matched_dataset:
            lines += [f"│  dataset       {self.matched_dataset}"]

        if self.language and self.language != "English":
            lines += [f"│  language      {self.language}"]

        if self.additional_notes:
            lines += [f"│  notes         {self.additional_notes}"]

        lines += [
            "│",
            "│  constraints",
            _bullet(self.constraints),
            "│",
            "│  key_skills",
            _bullet(self.key_skills),
            "│",
            "│  corner_cases",
            corner,
            "╰" + "─" * 62,
        ]

        return "\n".join(lines)


class GenerationResult(BaseModel):
    dataset: list[str]
    target: list[str]
    spec: TaskSpec | None = None
    description: str | None = None
