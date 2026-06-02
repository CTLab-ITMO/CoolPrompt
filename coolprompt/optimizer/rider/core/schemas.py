"""Structured schemas used by RIDER Genesis Ultra."""

from __future__ import annotations

import json
import re
from typing import Any, List

from pydantic import BaseModel, Field, field_validator


class _PromptContractSchema(BaseModel):
    """Instructor/Pydantic schema for the RIDER Genesis task contract."""

    task_archetype: str = "other"
    language: str = "unknown"
    domain: str = "general"
    audience: str = "general"
    output_format_anchor: str = "free-form"
    must_preserve: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    recommended_strategies: List[str] = Field(default_factory=list)
    avoid_strategies: List[str] = Field(default_factory=list)

    @field_validator("must_preserve", "failure_modes", "recommended_strategies", "avoid_strategies", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]


class _SyntheticTestsSchema(BaseModel):
    """Instructor schema for synthetic evaluation tests."""

    tests: List[str] = Field(default_factory=list, min_length=1)

    @field_validator("tests", mode="before")
    @classmethod
    def _coerce_tests(cls, value: Any) -> List[str]:
        if isinstance(value, dict):
            value = value.get("tests") or value.get("items") or value.get("inputs") or []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []
        tests: List[str] = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("input") or item.get("text") or item.get("case") or item
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            if text.strip():
                tests.append(text.strip())
        return tests


class _RedTeamAdversarialSchema(BaseModel):
    """Instructor schema for Ultra red-team findings."""

    edge_cases: List[str] = Field(default_factory=list)
    severity: str = "low"
    fix_directives: List[str] = Field(default_factory=list)

    @field_validator("edge_cases", "fix_directives", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]


class _JudgeScoreSchema(BaseModel):
    """Instructor schema for synthetic judge scores."""

    score: int = Field(ge=1, le=10)

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> int:
        if isinstance(value, str):
            match = re.search(r"\b(10|[1-9])\b", value)
            if match:
                return int(match.group(1))
        return int(value)
