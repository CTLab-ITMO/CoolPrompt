from typing import List
from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Response schema for a single prompt-improvement recommendation."""

    recommendation: str = Field(
        description=(
            "ONE general, universal recommendation for improving a prompt — "
            "no task-specific details, no references to particular failed "
            "examples. Strict requirements: "
            "(1) at most 20–25 words; "
            "(2) starts with an imperative action verb "
            "(e.g., 'Add', 'Specify', 'Use', 'Avoid', 'Include'); "
            "(3) no meta-comments such as 'similar to before', "
            "'as previously mentioned', 'this prompt', 'the recommendation "
            "is...'; "
            "(4) plain text only — no XML tags, no enumeration prefix, "
            "no quotes, no trailing explanation."
        )
    )


class FilteredRecommendationsResponse(BaseModel):
    """Response schema for the filtered/clustered recommendations list."""

    recommendations: List[str] = Field(
        description=(
            "A deduplicated list of synthesized recommendations produced by "
            "clustering semantically similar input recommendations and "
            "creating ONE NEW recommendation per cluster that captures the "
            "essence of the entire cluster — do NOT copy any input "
            "recommendation verbatim. "
            "The list MUST be ordered by cluster size in DESCENDING order "
            "(the recommendation derived from the largest cluster first). "
            "Each item follows the same single-recommendation rules: "
            "max 20–25 words, starts with an imperative action verb, "
            "no meta-comments, plain text only, no XML tags."
        )
    )
