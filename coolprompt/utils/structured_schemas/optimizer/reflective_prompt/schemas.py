from typing import List
from pydantic import BaseModel, Field


class InitialPromptResponse(BaseModel):
    """Response schema for initial-prompt generation from a problem description."""

    prompt: str = Field(
        description=(
            "A single, well-formed prompt that effectively solves the described task. "
            "Plain text, no XML tags (no <prompt></prompt>), no surrounding "
            "commentary, no explanations."
        )
    )


class ParaphrasedPromptsResponse(BaseModel):
    """Response schema for paraphrasing the initial prompt into a population."""

    prompts: List[str] = Field(
        description=(
            "List of paraphrased variants of the original prompt. "
            "Each variant must preserve the original meaning and language while "
            "differing in wording, sentence structure and/or tone. "
            "The list length MUST be exactly equal to the requested population "
            "size — no more, no fewer items. Each item is plain prompt text "
            "without XML tags or commentary."
        )
    )


class ShortTermHintResponse(BaseModel):
    """Response schema for short-term reflection hints (verbal gradient)."""

    hint: str = Field(
        description=(
            "ONE concise hint (strictly under 20 words) acting as a verbal "
            "gradient in prompt space: by comparing the worse and better "
            "prompt versions, suggest a single concrete edit operation that "
            "would push a prompt from worse to better. Prefer operations such "
            "as word replacement, conversion to active or positive voice, "
            "adding a missing word, or deleting a redundant word. "
            "Plain text only, no XML tags, no enumeration, no preamble."
        )
    )


class LongTermHintResponse(BaseModel):
    """Response schema for long-term reflection hints."""

    hint: str = Field(
        description=(
            "ONE constructive hint (strictly under 50 words) consolidating "
            "the previous long-term reflection together with the newly "
            "gathered short-term hints from this epoch into accumulated, "
            "epoch-spanning guidance for future prompt edits. "
            "Plain text, no XML tags, no enumeration, no preamble."
        )
    )


class CrossoverPromptResponse(BaseModel):
    """Response schema for crossover-stage improved prompts."""

    prompt: str = Field(
        description=(
            "An improved prompt for the same task, written by applying the "
            "provided short-term reflection hint. Output the new prompt as "
            "plain text only — no XML tags, no explanations, no commentary."
        )
    )


class MutatedPromptResponse(BaseModel):
    """Response schema for mutation-stage prompts."""

    prompt: str = Field(
        description=(
            "A mutated prompt for the same task, derived from the elitist "
            "(best-so-far) prompt according to the long-term reflection. "
            "Plain text only — no XML tags, no explanations, no commentary."
        )
    )
