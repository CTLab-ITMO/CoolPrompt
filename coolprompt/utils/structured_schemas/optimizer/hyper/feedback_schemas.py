from typing import List, Literal

from pydantic import BaseModel, Field


class SectionRecommendationResponse(BaseModel):
    """Response schema for a single section-targeted recommendation
    (used for both regular and contrastive feedback).

    Detailed style rules and section-naming guidance are provided by the
    feedback prompt template (see ``FEEDBACK_PROMPT_TEMPLATE`` /
    ``CONTRASTIVE_FEEDBACK_PROMPT`` in
    ``coolprompt.utils.prompt_templates.hyper_templates``). The schema
    only describes field semantics so as not to bias the model toward
    a particular section choice.
    """

    section: str = Field(
        description=(
            "Target section name for the recommendation. Either "
            "'general' (cross-cutting recommendation that applies "
            "across the whole prompt) or one of the section names "
            "listed in the prompt instructions."
        )
    )
    text: str = Field(
        description=(
            "ONE concise recommendation for improving the prompt. "
            "Plain text only — no XML tags, no enumeration prefix, "
            "no quotes, no trailing explanation. Style and content "
            "rules (action verb, length, anti-overfitting, etc.) "
            "are specified by the feedback prompt body."
        )
    )


class RecommendationGroupsResponse(BaseModel):
    """Response schema for grouping recommendations by semantic similarity."""

    groups: List[List[int]] = Field(
        description=(
            "Partition of the input recommendation ids into groups by "
            "semantic similarity. Each inner list contains the "
            "zero-based ids of items that belong to the same group. "
            "Singleton groups are allowed. Detailed grouping rules "
            "are specified by the feedback prompt body."
        )
    )


class SynthesizedRecommendationItem(BaseModel):
    """A single synthesized recommendation derived from a cluster."""

    text: str = Field(
        description=(
            "ONE concise synthesized recommendation that captures the "
            "essence of the corresponding group's members. Plain text "
            "only — no XML tags, no quotes, no enumeration prefix. "
            "Style rules are specified by the feedback prompt body."
        )
    )
    weight: int = Field(
        ge=1,
        description=(
            "Integer weight (>=1) reflecting the number of original "
            "recommendations represented by this synthesized item."
        ),
    )


class SynthesizedRecommendationsResponse(BaseModel):
    """Response schema for the per-section synthesis/filter step."""

    synthesized: List[SynthesizedRecommendationItem] = Field(
        description=(
            "Filtered, synthesized recommendations for a single "
            "section. Selection, ranking, conflict resolution and "
            "the cap on the number of items are specified by the "
            "feedback prompt body."
        )
    )


class InstanceLeakVerdict(BaseModel):
    """A single audit verdict for one recommendation."""

    verdict: Literal["KEEP", "REWRITE", "DROP"] = Field(
        description=(
            "Audit verdict for the recommendation: 'KEEP' (broad, "
            "actionable, applies to the problem type), 'REWRITE' "
            "(useful but contains instance-specific details that "
            "must be generalized — provide the rewrite in 'text'), "
            "or 'DROP' (no useful actionable change remains, or "
            "the recommendation is only vague/general)."
        )
    )
    text: str = Field(
        default="",
        description=(
            "Rewritten generalized recommendation. REQUIRED only "
            "when verdict == 'REWRITE'. For 'KEEP' and 'DROP', use "
            "an empty string."
        ),
    )


class InstanceLeakAuditResponse(BaseModel):
    """Response schema for the instance-leak audit pass over recommendations."""

    verdicts: List[InstanceLeakVerdict] = Field(
        description=(
            "List of audit verdicts, exactly ONE per input "
            "recommendation and in the SAME ORDER as the input. "
            "The list length MUST match the number of input "
            "recommendations."
        )
    )
