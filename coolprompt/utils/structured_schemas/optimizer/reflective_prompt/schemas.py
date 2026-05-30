from typing import List
from pydantic import BaseModel, Field


class InitialPromptResponse(BaseModel):
    """Response schema for initial-prompt generation from a problem description."""

    prompt: str = Field(
        description="A prompt that effectively solves the described task."
    )


class ParaphrasedPromptsResponse(BaseModel):
    """Response schema for paraphrasing the initial prompt into a population."""

    prompts: List[str] = Field(
        description=(
            "New variations of the original prompt keeping its initial meaning."
        )
    )


class ShortTermHintResponse(BaseModel):
    """Response schema for short-term reflection hints."""

    hint: str = Field(
        description=(
            "One small hint for designing better prompts, based on the two "
            "prompt versions, using less than 20 words."
        )
    )


class LongTermHintResponse(BaseModel):
    """Response schema for long-term reflection hints."""

    hint: str = Field(
        description=(
            "One constructive hint for designing better prompts, based on "
            "prior reflections and new insights, using less than 50 words."
        )
    )


class CrossoverPromptResponse(BaseModel):
    """Response schema for crossover-stage improved prompts."""

    prompt: str = Field(
        description=(
            "An improved prompt for the task, written according to the reflection."
        )
    )


class MutatedPromptResponse(BaseModel):
    """Response schema for mutation-stage prompts."""

    prompt: str = Field(
        description=(
            "A mutated prompt for the task, written according to the prior reflection."
        )
    )
