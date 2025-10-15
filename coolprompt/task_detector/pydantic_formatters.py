from pydantic import BaseModel, Field


class TaskDetectionStructuredOutputSchema(BaseModel):
    task: str = Field(
        task="Determined task classification"
    )
