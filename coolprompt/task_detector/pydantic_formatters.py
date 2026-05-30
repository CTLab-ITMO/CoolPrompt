"""Backward-compatibility shim.

The canonical schema now lives in
:mod:`coolprompt.utils.structured_schemas.task_detector`, alongside the
structured-output schemas used by the optimizer modules. This module
re-exports it under the historical name so external imports keep working.
"""

from coolprompt.utils.structured_schemas.task_detector import (
    TaskDetectionResponse as TaskDetectionStructuredOutputSchema,
)

__all__ = ["TaskDetectionStructuredOutputSchema"]
