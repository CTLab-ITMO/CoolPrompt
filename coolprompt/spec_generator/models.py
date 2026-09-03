"""Validated models for synthetic-data generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coolprompt.utils.enums import Task


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _normalize(values: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Trim values and remove empty case-insensitive duplicates."""

    if values is None:
        return None

    unique: dict[str, str] = {}
    for item in values:
        value = item.strip()
        if value:
            unique.setdefault(value.casefold(), value)

    return tuple(unique.values())


class Example(StrictModel):
    """One generated/seed example with optional alternative valid outputs.

    ``output`` is the primary target used by legacy code. ``references`` contains
    additional valid targets for the same input. Keeping one primary output preserves
    backward compatibility while allowing multi-reference metrics (e.g. CommonGen).
    """

    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    references: tuple[str, ...] = ()

    @model_validator(mode="after")
    def normalize_references(self) -> "Example":
        primary = self.output.strip().casefold()
        unique: dict[str, str] = {}
        for item in self.references:
            value = str(item).strip()
            if value and value.casefold() != primary:
                unique.setdefault(value.casefold(), value)
        object.__setattr__(self, "references", tuple(unique.values()))
        return self

    @property
    def all_outputs(self) -> tuple[str, ...]:
        """Return primary output followed by alternative valid outputs."""
        return (self.output, *self.references)


class TaskSpec(StrictModel):
    """Complete contract for generation and validation."""

    task: Task
    description: str = Field(min_length=1)
    input_format: str = Field(min_length=1)
    output_format: str = Field(min_length=1)
    requirements: tuple[str, ...] = ()
    labels: tuple[str, ...] | None = None
    language: str = Field(default="English", min_length=1)
    corner_cases: tuple[str, ...] = ()

    @field_validator(
        "requirements",
        "labels",
        "corner_cases",
    )
    @classmethod
    def normalize_collections(
            cls,
            values: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Normalize collection fields."""

        return _normalize(values)

    @model_validator(mode="after")
    def validate_labels(self) -> "TaskSpec":
        """Validate label usage for the selected task type."""

        is_classification = self.task == Task.CLASSIFICATION

        if is_classification and not self.labels:
            raise ValueError("Classification tasks require at least one label.")

        if not is_classification and self.labels is not None:
            raise ValueError("Labels are only valid for classification tasks.")

        return self


class TaskSpecDraft(BaseModel):
    """Optional overrides for an inferred TaskSpec."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task: Task | None = None
    description: str | None = Field(default=None, min_length=1)
    input_format: str | None = Field(default=None, min_length=1)
    output_format: str | None = Field(default=None, min_length=1)
    requirements: tuple[str, ...] | None = None
    labels: tuple[str, ...] | None = None
    language: str | None = Field(default=None, min_length=1)
    corner_cases: tuple[str, ...] | None = None

    @property
    def is_empty(self) -> bool:
        """Return whether no override fields were provided."""

        return not self.model_fields_set

    def overrides(self) -> dict[str, Any]:
        """Return explicitly provided override fields."""

        return self.model_dump(exclude_unset=True)


class GenerationContext(StrictModel):
    """Context shared across generation stages."""

    spec: TaskSpec
    dataset_name: str | None = None
    seed_examples: tuple[Example, ...] = ()


class GenerationResult(StrictModel):
    """Final generated dataset."""

    examples: tuple[Example, ...]
    context: GenerationContext

    @property
    def dataset(self) -> list[str]:
        """Return generated input values."""

        return [example.input for example in self.examples]

    @property
    def target(self) -> list[str]:
        """Return generated output values."""

        return [example.output for example in self.examples]

    @property
    def multireference_target(self) -> list[list[str]]:
        """Return all valid outputs per generated input for multi-reference metrics."""

        return [list(example.all_outputs) for example in self.examples]
