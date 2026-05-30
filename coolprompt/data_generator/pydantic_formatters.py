"""Backward-compatibility shim.

The synthetic-data-generator Pydantic schemas have been moved to
:mod:`coolprompt.utils.structured_schemas.data_generator` to align with
the project-wide convention used by :mod:`coolprompt.optimizer`.

This module preserves the original import paths and class names so that
existing user code and tests keep working. Prefer importing from
``coolprompt.utils.structured_schemas.data_generator`` in new code.
"""

from coolprompt.utils.structured_schemas.data_generator import (
    ClassificationTaskExample,
    ClassificationTaskResponse,
    GenerationTaskExample,
    GenerationTaskResponse,
    ProblemDescriptionResponse,
)

# Legacy aliases ------------------------------------------------------------
ProblemDescriptionStructuredOutputSchema = ProblemDescriptionResponse
ClassificationTaskStructuredOutputSchema = ClassificationTaskResponse
GenerationTaskStructuredOutputSchema = GenerationTaskResponse


__all__ = [
    "ClassificationTaskExample",
    "ClassificationTaskResponse",
    "ClassificationTaskStructuredOutputSchema",
    "GenerationTaskExample",
    "GenerationTaskResponse",
    "GenerationTaskStructuredOutputSchema",
    "ProblemDescriptionResponse",
    "ProblemDescriptionStructuredOutputSchema",
]
