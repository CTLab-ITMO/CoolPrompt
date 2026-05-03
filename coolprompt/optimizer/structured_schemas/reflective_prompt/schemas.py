"""Structured-output schemas for ReflectivePrompt LLM calls."""

from typing import List

from pydantic import BaseModel, Field


class InitialPromptResponse(BaseModel):
    """Response schema for initial-prompt generation from a problem description."""

    prompt: str = Field(
        description=(
            "A single, well-formed prompt that effectively solves the described task. "
            "Plain text, no XML tags, no surrounding commentary."
        )
    )


class ParaphrasedPromptsResponse(BaseModel):
    """Response schema for paraphrasing the initial prompt into a population."""

    prompts: List[str] = Field(
        description=(
            "List of paraphrased variants of the original prompt. "
            "Each variant must preserve the original meaning while differing in "
            "wording and structure. The list length must match the requested "
            "population size."
        )
    )


class ShortTermHintResponse(BaseModel):
    """Response schema for short-term reflection hints."""

    hint: str = Field(
        description=(
            "One concise hint (under 20 words) for designing better prompts, "
            "based on comparing a worse and a better prompt version. "
            "Plain text, no XML tags."
        )
    )


class LongTermHintResponse(BaseModel):
    """Response schema for long-term reflection hints."""

    hint: str = Field(
        description=(
            "One constructive hint (under 50 words) consolidating prior reflections "
            "and newly gained short-term insights. Plain text, no XML tags."
        )
    )


class CrossoverPromptResponse(BaseModel):
    """Response schema for crossover-stage improved prompts."""

    prompt: str = Field(
        description=(
            "An improved prompt produced by crossover, based on a worse prompt, "
            "a better prompt and a short-term reflection hint. Plain text only."
        )
    )


class MutatedPromptResponse(BaseModel):
    """Response schema for mutation-stage prompts."""

    prompt: str = Field(
        description=(
            "A mutated prompt derived from the elitist prompt and the long-term "
            "reflection. Plain text only, no XML tags or commentary."
        )
    )
