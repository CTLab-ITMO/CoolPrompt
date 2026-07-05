from pydantic import BaseModel, Field

SUPPORTED_TASK_AREAS = [
    "tweet_emotional_classification",
    "school_math_reasoning",
    "concept_to_sentence_generation",
    "context_question_answering",
    "text_summarization",
]


class TaskDetectionStructuredOutputSchema(BaseModel):
    """Structured response containing the detected CoolPrompt task type."""

    task: str = Field(description="Determined task classification")


class TaskAreaDetectionStructuredOutputSchema(BaseModel):
    task: str = Field(
        description="Detected task type. Usually 'classification' or 'generation'."
    )

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
