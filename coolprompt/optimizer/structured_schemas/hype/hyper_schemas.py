from pydantic import BaseModel, Field


class ParaphrasedVariantResponse(BaseModel):
    """Response schema for HyPER paraphrasing of the current best prompt."""

    paraphrased_prompt: str = Field(
        description=(
            "An alternative version of the original prompt that preserves "
            "meaning, key details and language, but uses different words, "
            "structure or tone. Plain text only."
        )
    )
