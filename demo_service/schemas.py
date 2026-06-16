"""Pydantic schemas for the CoolPrompt Interface Demo API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TaskName = Literal["classification", "generation"]


class OptimizationRequest(BaseModel):
    """A single method optimization request."""

    start_prompt: str = Field(min_length=1, max_length=20000)
    task: TaskName = "generation"
    method: str = "hyper_light"
    metric: str | None = None
    problem_description: str | None = Field(default=None, max_length=4000)
    dataset: list[str] | None = None
    target: list[str | int] | None = None
    validation_size: float = Field(default=0.34, ge=0.0, le=1.0)
    train_as_test: bool = False
    generate_num_samples: int = Field(default=6, ge=2, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    verbose: int = Field(default=1, ge=0, le=2)
    model_name: str | None = Field(default=None, max_length=120)
    model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    model_max_tokens: int = Field(default=2000, ge=128, le=16000)
    method_params: dict[str, Any] = Field(default_factory=dict)
    mock: bool = False

    @field_validator("dataset", "target", mode="before")
    @classmethod
    def empty_lists_to_none(cls, value):
        if value == []:
            return None
        return value

    @model_validator(mode="after")
    def validate_dataset_pairs(self):
        if self.dataset is None and self.target is None:
            return self
        if self.dataset is None or self.target is None:
            raise ValueError("dataset and target must be provided together")
        if len(self.dataset) != len(self.target):
            raise ValueError("dataset and target must have the same length")
        if len(self.dataset) < 2:
            raise ValueError("dataset must contain at least 2 rows")
        return self


class CompareRequest(BaseModel):
    """Run several methods on the same input."""

    base: OptimizationRequest
    methods: list[str] = Field(min_length=1, max_length=7)
    method_params_by_method: dict[str, dict[str, Any]] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    """Create either a single optimization job or a comparison job."""

    mode: Literal["single", "compare"] = "single"
    request: OptimizationRequest | None = None
    compare: CompareRequest | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.mode == "single" and self.request is None:
            raise ValueError("request is required for single mode")
        if self.mode == "compare" and self.compare is None:
            raise ValueError("compare is required for compare mode")
        return self


class OptimizationResult(BaseModel):
    """Optimization output returned to the browser."""

    method: str
    initial_prompt: str
    final_prompt: str
    init_metric: float | None = None
    final_metric: float | None = None
    metric_delta: float | None = None
    dataset_size: int
    validation_size: int
    elapsed_seconds: float
    used_mock: bool = False
    model_name: str | None = None
    method_params: dict[str, Any] = Field(default_factory=dict)
    synthetic_dataset: list[str] | None = None
    synthetic_target: list[str | int] | None = None


class JobStatus(BaseModel):
    """In-memory job state."""

    job_id: str
    mode: Literal["single", "compare"]
    status: Literal["queued", "running", "completed", "failed"]
    created_at: float
    updated_at: float
    error: str | None = None
    result: OptimizationResult | list[OptimizationResult] | None = None
