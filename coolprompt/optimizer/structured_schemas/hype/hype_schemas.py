from pydantic import BaseModel, Field


class OptimizedPromptResponse(BaseModel):
    """Response schema for HyPE-generated optimized prompt."""

    result_prompt: str = Field(
        description=(
            "The final optimized prompt produced according to the HyPE "
            "meta-prompt structure. Return the prompt content only, without "
            "wrapping XML tags or extra commentary."
        )
    )
