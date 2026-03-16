"""Pydantic schemas for PE2+SGR structured reasoning."""

from enum import Enum
from typing import List, Optional, Literal

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
        description="Brief description of the input that caused the error"
    )
    expected_vs_actual: str = Field(
        description="Comparison of expected and actual output"
    )
    root_cause: ErrorType = Field(
        description="Category of the root cause of the error"
    )
    root_cause_explanation: str = Field(
        description="Detailed explanation of why the error occurred"
    )


class PromptAnalysis(BaseModel):
    """Step 1: Analysis of task description correctness."""

    describes_task_correctly: bool = Field(
        description="Does the prompt correctly describe the task?"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of missing elements in the task description"
    )
    misleading_elements: List[str] = Field(
        default_factory=list,
        description="List of potentially misleading instructions"
    )


class EditDecision(BaseModel):
    """Step 2: Decision on whether editing is necessary."""

    editing_necessary: bool = Field(
        description="Is editing of the prompt necessary?"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level in the editing decision"
    )
    justification: str = Field(
        description="Justification for the editing decision"
    )


class PromptChange(BaseModel):
    """A single specific change to the prompt."""

    change_type: Literal[
        "add", "remove", "modify", "restructure"
    ] = Field(
        description="Type of change"
    )
    location: str = Field(
        description="Where in the prompt to make the change"
    )
    original_text: Optional[str] = Field(
        default=None,
        description=(
            "Original text to change (if applicable)"
        ),
    )
    new_text: str = Field(
        description="New text to insert or replace with"
    )
    rationale: str = Field(
        description=(
            "Why this change addresses the identified issues"
        ),
    )


class PE2SGROutput(BaseModel):
    """Full structured output for PE2+SGR."""

    error_analyses: List[ErrorAnalysis] = Field(
        min_length=1,
        description=(
            "Analysis of each failure example from the batch"
        ),
    )

    prompt_analysis: PromptAnalysis

    edit_decision: EditDecision

    specific_changes: List[PromptChange] = Field(
        default_factory=list,
        description="List of specific changes to make"
    )

    improved_prompt: str = Field(
        description=(
            "The full improved prompt after applying "
            "all changes"
        ),
    )

    improvement_summary: str = Field(
        description=(
            "One-sentence summary of the main improvement"
        ),
    )
