from pydantic import BaseModel, Field
from typing import List


class ProblemDescriptionStructuredOutputSchema(BaseModel):
    """Structured response containing a generated problem description."""

    problem_description: str = Field(
        description="Determined problem description"
    )


class ClassificationTaskExample(BaseModel):
    """Single synthetic classification sample."""

    input: str = Field(description="Input request")
    output: str = Field(description="Output label")


class ClassificationTaskStructuredOutputSchema(BaseModel):
    """Structured response containing classification examples."""

    examples: List[ClassificationTaskExample] = Field(
        description="List of examples like "
        + '{"input": "...", "output": "ground-truth label"}'
    )


class GenerationTaskExample(BaseModel):
    """Single synthetic generation sample."""

    input: str = Field(description="Input request")
    output: str = Field(description="LLM answer")


class GenerationTaskStructuredOutputSchema(BaseModel):
    """Structured response containing generation examples."""

    examples: List[GenerationTaskExample] = Field(
        description='List of examples like {"input": "...", "output": "..."}'
    )
