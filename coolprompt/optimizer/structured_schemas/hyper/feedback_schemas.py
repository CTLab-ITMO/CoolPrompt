from typing import List

from pydantic import BaseModel, Field


class SectionRecommendationResponse(BaseModel):
    """Response schema for a single section-targeted recommendation
    (used for both regular and contrastive feedback)."""

    section: str = Field(
        description=(
            "Target section name for the recommendation. MUST be either "
            "'general' (cross-cutting recommendation) or one of the "
            "section names listed in the prompt's '### STRUCTURE OF THE "
            "PROMPT YOU MUST PRODUCE' block (typically 'Role', "
            "'Task context', 'Instructions', 'Output requirements'). "
            "Prefer 'general' when unsure."
        )
    )
    text: str = Field(
        description=(
            "ONE concise recommendation for improving the prompt. Strict "
            "requirements: "
            "(1) at most 20-25 words; "
            "(2) starts with an imperative action verb (e.g. 'Add', "
            "'Specify', 'Use', 'Require', 'Avoid', 'Include'); "
            "(3) NO task-specific details and NO references to particular "
            "failed examples (no named entities, formulas, scenarios or "
            "dataset-specific phrases that would not generalize to other "
            "failures of the same task class); "
            "(4) NO meta-comments such as 'similar to before', "
            "'as previously mentioned', 'this prompt', "
            "'the recommendation is...'; "
            "(5) plain text only — no XML tags, no enumeration prefix, "
            "no quotes, no trailing explanation."
        )
    )


class RecommendationGroupsResponse(BaseModel):
    """Response schema for grouping recommendations by semantic similarity."""

    groups: List[List[int]] = Field(
        description=(
            "Partition of the input recommendation ids into groups by "
            "semantic similarity. Each inner list contains the zero-based "
            "ids of items that belong to the same group. "
            "Rules: "
            "(1) items addressing different aspects MUST go to different "
            "groups; "
            "(2) singleton groups are allowed if an item has no similar "
            "peer; "
            "(3) each input id MUST appear EXACTLY ONCE across all "
            "groups — no duplicates, no omissions; "
            "(4) ids must reference valid positions in the input list."
        )
    )


class SynthesizedRecommendationItem(BaseModel):
    """A single synthesized recommendation derived from a cluster."""

    text: str = Field(
        description=(
            "ONE concise synthesized recommendation that captures the "
            "essence of the corresponding group's members. Style: starts "
            "with an action verb, max 20-25 words, no task-specific "
            "details, no meta-comments ('as before', 'similar to'). "
            "Plain text only — no XML tags, no quotes, no enumeration "
            "prefix."
        )
    )
    weight: int = Field(
        ge=1,
        description=(
            "Integer weight (>=1) equal to the number of original "
            "recommendations in the source group that this synthesized "
            "recommendation represents."
        ),
    )


class SynthesizedRecommendationsResponse(BaseModel):
    """Response schema for the per-section synthesis/filter step."""

    synthesized: List[SynthesizedRecommendationItem] = Field(
        description=(
            "Filtered, synthesized recommendations for a single section. "
            "Rules: "
            "(1) produce ONE synthesized item per surviving group; "
            "(2) resolve mutually contradictory groups by keeping the one "
            "with the larger weight (or the more useful one when weights "
            "tie); "
            "(3) return AT MOST 3 items total — prioritize concrete "
            "input/output constraints, decision rules and verification "
            "steps over tone/style/quality preferences; "
            "(4) do not keep recommendations that depend on seeing the "
            "reference/correct answer at inference time."
        )
    )


class InstanceLeakVerdict(BaseModel):
    """A single audit verdict for one recommendation."""

    verdict: str = Field(
        description=(
            "Audit verdict for the recommendation. MUST be one of: "
            "'KEEP' (broad, actionable, applies to the problem type), "
            "'REWRITE' (useful prompt-level change but mentions a narrow "
            "subtype/formula/entity/scenario/dataset-specific phrase or "
            "reference-answer leak — provide a generalized rewrite in "
            "'text'), or 'DROP' (no useful actionable change remains, or "
            "the recommendation is only vague/general)."
        )
    )
    text: str = Field(
        default="",
        description=(
            "Rewritten generalized recommendation. REQUIRED only when "
            "verdict == 'REWRITE'. For 'KEEP' and 'DROP', use an empty "
            "string. Same style rules as other recommendations: action "
            "verb, max 20-25 words, no narrow entities/formulas/scenarios."
        ),
    )


class InstanceLeakAuditResponse(BaseModel):
    """Response schema for the instance-leak audit pass over recommendations."""

    verdicts: List[InstanceLeakVerdict] = Field(
        description=(
            "List of audit verdicts, exactly ONE per input recommendation "
            "and in the SAME ORDER as the input. The list length MUST "
            "match the number of input recommendations."
        )
    )
