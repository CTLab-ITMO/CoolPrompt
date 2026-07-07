from pydantic import BaseModel, Field


class TaskDetectionStructuredOutputSchema(BaseModel):
    """Structured response containing the detected CoolPrompt task type."""

    task: str = Field(description="Determined task classification")
