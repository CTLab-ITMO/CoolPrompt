from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.task_detector.pydantic_formatters import (
    TaskDetectionStructuredOutputSchema,
    TaskAreaDetectionStructuredOutputSchema,
)
from coolprompt.utils.prompt_templates.task_detector_templates import (
    TASK_DETECTOR_TEMPLATE,
    TASK_AREA_DETECTOR_TEMPLATE,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


class TaskDetector:
    """Task Detector
    Defines task problem for prompt optimization

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
    """

    def __init__(self, model: BaseLanguageModel, confidence_threshold: float = 0.7) -> None:
        self.model = model
        self._confidence_threshold = confidence_threshold

    def _generate(
            self, request: str, schema: BaseModel, field_name: str
    ) -> Any:
        """Generates model output
        either using structured output from langchain
        or just strict json output format for LLM

        Args:
            request (str): request to LLM
                when langchain structured output is used
            schema (BaseModel): Pydantic output format
            field_name (str): field name to select from output

        Returns:
            Any: generated data
        """
        if hasattr(self.model, "model"):
            wrapped_model = self.model.model
        else:
            wrapped_model = self.model

        if not isinstance(wrapped_model, BaseChatModel):
            output = self.model.invoke(request)
            if isinstance(output, AIMessage):
                output = output.content
            return extract_json(output)[field_name]

        structured_model = self.model.with_structured_output(
            schema=schema, method="json_schema"
        )
        output = structured_model.invoke(request)
        if isinstance(output, AIMessage):
            output = output.content

        try:
            output = getattr(output, field_name)
        except Exception:
            output = output[field_name]
        return output

    def generate(
            self,
            prompt: str,
    ) -> str:
        """Defines task definition

        Args:
            prompt (str): initial user prompt

        Returns:
            str: task class
        """
        schema = TaskDetectionStructuredOutputSchema
        request = TASK_DETECTOR_TEMPLATE

        request = request.format(query=prompt)

        logger.info("Detecting the task by query")

        task = self._generate(request, schema, "task")

        logger.info(f"Task defined as {task}")

        return task

    def _generate_structured(self, request: str, schema: type[BaseModel]) -> Any:
        """Generates and validates structured model output.

        Args:
            request (str): request to LLM
            schema (type[BaseModel]): Pydantic output format

        Returns:
            BaseModel: validated structured response
        """
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

    def detect_task_area(self, prompt: str) -> TaskAreaDetectionStructuredOutputSchema:
        """Detects task type and supported task area.

        Args:
            prompt (str): initial user prompt

        Returns:
            TaskAreaDetectionStructuredOutputSchema: detected task area result
        """
        logger.info("Detecting task area by query")

        result = self._generate_structured(
            request=TASK_AREA_DETECTOR_TEMPLATE.format(query=prompt),
            schema=TaskAreaDetectionStructuredOutputSchema,
        )

        if result.confidence < self._confidence_threshold:
            logger.info(
                "Task area confidence too low: area=%r, confidence=%.2f (threshold=%.2f) — treating as unmatched",
                result.task_area,
                result.confidence,
                self._confidence_threshold,
            )
            return result.model_copy(update={"task_area": None})

        logger.info(
            "Task area detected: task=%s, area=%s, confidence=%.2f",
            result.task, result.task_area, result.confidence,
        )
        return result
