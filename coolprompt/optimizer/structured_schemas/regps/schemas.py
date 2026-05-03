from pydantic import BaseModel, Field


class TextualGradientResponse(BaseModel):
    """Response schema for textual-gradient feedback generation."""

    feedback: str = Field(
        description=(
            "Detailed, actionable feedback describing the prompt's weaknesses "
            "based on failed examples, and how it can be improved to avoid the "
            "same mistakes. Plain text only, no XML tags."
        )
    )
