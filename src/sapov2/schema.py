"""Pydantic schemas for SAPO v2 structured outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResponseTemplate(BaseModel):
    """Schema for plain model outputs."""

    response: str = Field(..., description="Final model response")


class ReasoningTemplate(BaseModel):
    """Schema for reasoning-enabled outputs."""

    reasoning: str = Field(..., description="Reasoning trace")
    response: str = Field(..., description="Final model response")


class PromptSegments(BaseModel):
    """Prompt segmentation used by SAPO and SAPO v2."""

    role: str = Field(default="", description="Role instruction")
    context: str = Field(default="", description="Task context")
    tasks: str = Field(default="", description="Task instructions")
    output_format: str = Field(default="", description="Formatting constraints")


class WeaknessAnalysis(BaseModel):
    """Segment-level strengths, weaknesses and recommendations."""

    weak_segments: list[str] = Field(default_factory=list)
    strong_segments: list[str] = Field(default_factory=list)
    recommendations: dict[str, str] = Field(default_factory=dict)


class CandidateGenerationResponse(BaseModel):
    """Structured candidate generation response."""

    prompts: list[str] = Field(default_factory=list)


class SegmentConfidence(BaseModel):
    """Confidence scores for segments in [0, 1]."""

    role: float = 0.0
    context: float = 0.0
    tasks: float = 0.0
    output_format: float = 0.0


class ScoreBreakdown(BaseModel):
    """Multi-objective score components for candidate ranking."""

    bertscore: float
    format_compliance: float
    length_penalty: float
    estimated_cost: float
    objective: float
