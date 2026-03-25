"""Pydantic schemas for PE2+SGR v3 structured reasoning.

Above the score threshold, Phase 1 produces a structured
FullDiagnosis with strategy override.  Below the threshold,
Phase 1 uses free-form reasoning (no structured output).
Phase 2 always generates the improved prompt via free-form
text.
"""

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Categories of prompt errors."""

    TASK_UNCLEAR = "task_description_unclear"
    MISSING_CONSTRAINTS = "missing_constraints"
    WRONG_FORMAT = "wrong_output_format"
    INCOMPLETE_GUIDANCE = "incomplete_guidance"
    OVERSPECIFICATION = "overspecification"
    OTHER = "other"


class ErrorAnalysis(BaseModel):
    """Analysis of a single failure example."""

    input_summary: str = Field(
        description=(
            "Brief description of the input that "
            "caused the error"
        )
    )
    expected_vs_actual: str = Field(
        description=(
            "Comparison of expected and actual output"
        )
    )
    root_cause: ErrorType = Field(
        description=(
            "Category of the root cause of the error"
        )
    )
    root_cause_explanation: str = Field(
        description=(
            "Detailed explanation of why the error occurred"
        )
    )


class PromptAnalysis(BaseModel):
    """Analysis of task description correctness."""

    describes_task_correctly: bool = Field(
        description=(
            "Does the prompt correctly describe the task?"
        )
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description=(
            "List of missing elements in the task "
            "description"
        ),
    )
    misleading_elements: List[str] = Field(
        default_factory=list,
        description=(
            "List of potentially misleading instructions"
        ),
    )


class EditDecision(BaseModel):
    """Decision on whether editing is necessary."""

    editing_necessary: bool = Field(
        description=(
            "Is editing of the prompt necessary?"
        )
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence level in the editing decision"
        )
    )
    justification: str = Field(
        description=(
            "Justification for the editing decision"
        )
    )


class PatternSynthesis(BaseModel):
    """Cross-example failure pattern analysis."""

    common_failure_pattern: str = Field(
        description="What unifies all failure examples"
    )
    pattern_severity: Literal[
        "surface", "structural", "fundamental"
    ] = Field(
        description=(
            "surface=cosmetic/format issues, "
            "structural=missing key instructions, "
            "fundamental=entire approach is wrong"
        )
    )
    error_homogeneity: Literal[
        "low", "medium", "high"
    ] = Field(
        description=(
            "How similar are the root causes across "
            "examples. low=diverse errors, "
            "high=all same root cause"
        )
    )


class RewriteStrategy(BaseModel):
    """Explicit strategy selection for Phase 2."""

    approach: Literal[
        "incremental_edit",
        "structural_rewrite",
        "complete_reimagine",
    ] = Field(
        description=(
            "incremental_edit=fix specific issues, "
            "structural_rewrite=reorganize significantly, "
            "complete_reimagine=design from scratch"
        )
    )
    justification: str = Field(
        description="Why this strategy was chosen"
    )
    key_insight: str = Field(
        description=(
            "The single most important thing to change"
        )
    )


class FullDiagnosis(BaseModel):
    """Phase 1 schema when best_val_score >= threshold.

    Full per-example analysis plus global patterns.
    Used when task is partially solved and needs
    surgical precision.
    """

    error_analyses: List[ErrorAnalysis] = Field(
        min_length=1,
        description=(
            "Analysis of each failure example "
            "from the batch"
        ),
    )
    pattern_synthesis: PatternSynthesis
    prompt_analysis: PromptAnalysis
    edit_decision: EditDecision
    rewrite_strategy: RewriteStrategy
