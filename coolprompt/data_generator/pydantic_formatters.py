from pydantic import BaseModel, Field
from typing import List


class ProblemDescriptionStructuredOutputSchema(BaseModel):
    problem_description: str = Field(
        description="Determined problem description"
    )


class ClassificationTaskExample(BaseModel):
    input: str = Field(description="Input request")
    output: str = Field(description="Output label")


class ClassificationTaskStructuredOutputSchema(BaseModel):
    examples: List[ClassificationTaskExample] = Field(
        description="List of examples like "
        + '{"input": "...", "output": "ground-truth label"}'
    )


class GenerationTaskExample(BaseModel):
    input: str = Field(description="Input request")
    output: str = Field(description="LLM answer")


class GenerationTaskStructuredOutputSchema(BaseModel):
    examples: List[GenerationTaskExample] = Field(
        description='List of examples like {"input": "...", "output": "..."}'
    )
