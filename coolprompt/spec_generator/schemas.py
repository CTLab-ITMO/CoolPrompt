"""Structured-output schemas used by the generation model."""

from pydantic import BaseModel, Field


class TaskExample(BaseModel):
    """One generated input-output example."""

    input: str = Field(min_length=1, description="Example input")
    output: str = Field(description="Example output")


class TaskExamples(BaseModel):
    """A non-empty batch of generated examples."""

    examples: list[TaskExample] = Field(
        min_length=1,
        description="Generated synthetic examples",
    )


__all__ = ["TaskExample", "TaskExamples"]
