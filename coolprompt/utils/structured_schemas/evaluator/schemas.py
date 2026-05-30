from pydantic import BaseModel, Field


class ClassificationAnswerResponse(BaseModel):
    """Response schema for classification-task answers."""

    answer: str = Field(description="The chosen label for the given input.")


class GenerationAnswerResponse(BaseModel):
    """Response schema for free-form generation-task answers."""

    answer: str = Field(description="The final answer to the task.")


class JudgeScoreResponse(BaseModel):
    """Response schema for the LLM-as-a-judge metric score."""

    score: int = Field(description="Integer score for the requested criterion.")
