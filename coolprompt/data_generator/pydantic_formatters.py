from coolprompt.utils.structured_schemas.data_generator import (
    ClassificationTaskExample,
    ClassificationTaskResponse,
    GenerationTaskExample,
    GenerationTaskResponse,
    ProblemDescriptionResponse,
)

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
