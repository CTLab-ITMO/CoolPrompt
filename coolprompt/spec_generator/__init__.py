"""Synthetic-data specification and generation API."""

from .generator import SyntheticDataGenerator
from .models import (
    Example,
    GenerationContext,
    GenerationResult,
    TaskSpec,
    TaskSpecDraft,
)
from .prompt_builder import GenerationPromptBuilder
from .spec_builder import SpecBuilder
from .validation import Deduplicator, ExampleValidator, LLMJudge, ValidationPipeline

__all__ = [
    "Deduplicator",
    "Example",
    "ExampleValidator",
    "GenerationContext",
    "GenerationPromptBuilder",
    "GenerationResult",
    "LLMJudge",
    "SpecBuilder",
    "SyntheticDataGenerator",
    "TaskSpec",
    "TaskSpecDraft",
    "ValidationPipeline",
]
