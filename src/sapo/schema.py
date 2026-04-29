"""Pydantic schemas for structured SAPO LLM responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResponseTemplate(BaseModel):
    """Schema for plain responses without explicit reasoning."""

    response: str = Field(..., description="Final model response")


class ReasoningTemplate(BaseModel):
    """Schema for responses that include a reasoning trace."""

    reasoning: str = Field(..., description="Reasoning used to derive the final answer")
    response: str = Field(..., description="Final model response")


class ReasoningCandGenTemplate(BaseModel):
    """Schema for candidate prompt generation responses."""

    reasoning: str = Field(..., description="Reasoning used to produce prompt candidates")
    response: list[str] = Field(..., description="List of generated prompt candidates")
