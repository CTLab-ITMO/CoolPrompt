from pydantic import BaseModel, Field

from coolprompt.utils.task_areas import SUPPORTED_TASK_AREAS


class TaskDetectionStructuredOutputSchema(BaseModel):
    """Structured response containing the detected CoolPrompt task type."""

    task: str = Field(description="Determined task classification")


class TaskAreaDetectionStructuredOutputSchema(BaseModel):
    """Structured output for task area detection."""

    task: str = Field(description="Detected task type. Usually 'classification' or 'generation'.")

    task_area: str | None = Field(
        default=None,
        description=(
            "Detected task area. One of: "
            f"{', '.join(SUPPORTED_TASK_AREAS)}, "
            "or null if no supported area matches."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the selected task area.",
    )

    reason: str = Field(
        description="Short explanation of why this task area was selected."
    )
