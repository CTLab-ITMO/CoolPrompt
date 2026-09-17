from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.task_detector.pydantic_formatters import (
    TaskAreaDetectionStructuredOutputSchema,
    TaskDetectionStructuredOutputSchema,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.task_detector_templates import (
    TASK_AREA_DETECTOR_TEMPLATE,
    TASK_DETECTOR_TEMPLATE,
)


class TaskDetector:
    """Detect a task definition and supported task area from a user prompt."""

    def __init__(
        self,
        model: BaseLanguageModel,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.model = model
        self._confidence_threshold = confidence_threshold

    def _generate(self, request: str, schema: type[BaseModel], field_name: str) -> Any:
        """Generate model output and extract the requested response field."""
        wrapped_model = getattr(self.model, "model", self.model)

        if not isinstance(wrapped_model, BaseChatModel):
            output = self.model.invoke(request)
            if isinstance(output, AIMessage):
                output = output.content
            return extract_json(output)[field_name]

        output = self.model.with_structured_output(
            schema=schema,
            method="json_schema",
        ).invoke(request)
        if isinstance(output, AIMessage):
            output = output.content

        try:
            return getattr(output, field_name)
        except (AttributeError, TypeError):
            return output[field_name]

    def generate(self, prompt: str) -> str:
        """Return the task type detected from the user prompt."""
        logger.info("Detecting the task by query")
        task = self._generate(
            TASK_DETECTOR_TEMPLATE.format(query=prompt),
            TaskDetectionStructuredOutputSchema,
            "task",
        )
        logger.info("Task defined as %s", task)
        return task

    def _generate_structured(
        self,
        request: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Generate and validate structured model output."""
        wrapped_model = getattr(self.model, "model", self.model)

        if not isinstance(wrapped_model, BaseChatModel):
            output = self.model.invoke(request)
            content = output.content if isinstance(output, AIMessage) else str(output)
            return schema(**extract_json(content))

        output = self.model.with_structured_output(
            schema=schema,
            method="json_schema",
        ).invoke(request)

        if isinstance(output, dict):
            return schema(**output)
        if isinstance(output, AIMessage):
            return schema(**extract_json(output.content))
        if isinstance(output, schema):
            return output

        raise TypeError(f"Unexpected structured output type: {type(output)!r}")

    def detect_task_area(
        self,
        prompt: str,
    ) -> TaskAreaDetectionStructuredOutputSchema:
        """Detect the task type and supported task area."""
        logger.info("Detecting task area by query")
        result = self._generate_structured(
            request=TASK_AREA_DETECTOR_TEMPLATE.format(query=prompt),
            schema=TaskAreaDetectionStructuredOutputSchema,
        )

        if not isinstance(result, TaskAreaDetectionStructuredOutputSchema):
            raise TypeError(f"Unexpected task-area result type: {type(result)!r}")

        if result.confidence < self._confidence_threshold:
            logger.info(
                "Task area confidence too low: area=%r, confidence=%.2f "
                "(threshold=%.2f); treating as unmatched",
                result.task_area,
                result.confidence,
                self._confidence_threshold,
            )
            return result.model_copy(update={"task_area": None})

        logger.info(
            "Task area detected: task=%s, area=%s, confidence=%.2f",
            result.task,
            result.task_area,
            result.confidence,
        )
        return result
