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
from .schemas import TaskExample, TaskExamples
from .spec_builder import SpecBuilder
from .validation import Deduplicator, ExampleValidator, ValidationPipeline

__all__ = [
    "Deduplicator",
    "Example",
    "ExampleValidator",
    "GenerationContext",
    "GenerationPromptBuilder",
    "GenerationResult",
    "SpecBuilder",
    "SyntheticDataGenerator",
    "TaskSpec",
    "TaskSpecDraft",
    "TaskExample",
    "TaskExamples",
    "ValidationPipeline",
]
