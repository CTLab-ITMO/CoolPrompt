"""Structured-output schemas for HyPER FeedbackModule."""

from typing import List

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Response schema for a single prompt-improvement recommendation."""

    recommendation: str = Field(
        description=(
            "One general, universal recommendation to improve the prompt "
            "(no task-specific details). Concise, max 20-25 words, starts "
            "with an action verb. No meta-comments."
        )
    )


class FilteredRecommendationsResponse(BaseModel):
    """Response schema for the filtered/clustered recommendations list."""

    recommendations: List[str] = Field(
        description=(
            "Synthesized, deduplicated recommendations ordered by importance "
            "(largest cluster first). Each item is a single new recommendation "
            "capturing the essence of its cluster, not a verbatim copy."
        )
    )
