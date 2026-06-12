from pydantic import BaseModel, Field


class ResultPromptResponse(BaseModel):
    """Response schema for HyPER meta-prompt single-step optimization."""

    result_prompt: str = Field(
        description="The optimized prompt produced by the meta-prompt."
    )


class ParaphrasedVariantResponse(BaseModel):
    """Response schema for paraphrasing the current best HyPER prompt."""

    paraphrased_prompt: str = Field(
        description="A paraphrased variant of the input prompt."
    )
