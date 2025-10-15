from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.task_detector.pydantic_formatters import (
    TaskDetectionStructuredOutputSchema
)
from coolprompt.utils.prompt_templates.task_detector_templates import (
    TASK_DETECTOR_TEMPLATE
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


class TaskDetector:
    """Task Detector
    Defines task problem for prompt optimization

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
    """

    def __init__(self, model: BaseLanguageModel) -> None:
        self.model = model

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
        if not isinstance(self.model, BaseChatModel):
            output = self.model.invoke(request)
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

        request = request.format(
            query=prompt
        )

        logger.info(
            "Detecting the task by query"
        )

        task = self._generate(request, schema, "task")

        logger.info(
            f"Task defined as {task}"
        )

        return task
