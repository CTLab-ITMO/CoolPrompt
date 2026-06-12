from typing import List, Literal

from pydantic import BaseModel, Field


class SectionRecommendationResponse(BaseModel):
    """Response schema for a single section-targeted recommendation
    (used for both regular and contrastive feedback)."""

    section: str = Field(
        description="Target section name for the recommendation, or 'general'."
    )
    text: str = Field(
        description="The recommendation text."
    )


class RecommendationGroupsResponse(BaseModel):
    """Response schema for grouping recommendations by semantic similarity."""

    groups: List[List[int]] = Field(
        description=(
            "Partition of input recommendation ids into groups. "
            "Each inner list contains the zero-based ids belonging to one group."
        )
    )


class SynthesizedRecommendationItem(BaseModel):
    """A single synthesized recommendation derived from a cluster."""

    text: str = Field(
        description="The synthesized recommendation text for the group."
    )
    weight: int = Field(
        ge=1,
        description=(
            "Number of original recommendations represented by this synthesized item."
        ),
    )


class SynthesizedRecommendationsResponse(BaseModel):
    """Response schema for the per-section synthesis/filter step."""

    synthesized: List[SynthesizedRecommendationItem] = Field(
        description="Synthesized recommendations for the section."
    )


class InstanceLeakVerdict(BaseModel):
    """A single audit verdict for one recommendation."""

    verdict: Literal["KEEP", "REWRITE", "DROP"] = Field(
        description="Audit verdict for the recommendation."
    )
    text: str = Field(
        default="",
        description="Rewritten recommendation; used only when verdict is 'REWRITE'.",
    )


class InstanceLeakAuditResponse(BaseModel):
    """Response schema for the instance-leak audit pass over recommendations."""

    verdicts: List[InstanceLeakVerdict] = Field(
        description=(
            "One verdict per input recommendation, in the same order as the input."
        )
    )
