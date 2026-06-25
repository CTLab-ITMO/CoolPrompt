from typing import Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.utils.structured_schemas.task_detector import (
    TaskDetectionResponse,
)
from coolprompt.utils.prompt_templates.task_detector_templates import (
    TASK_DETECTOR_TEMPLATE,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


class TaskDetector:
    """Task Detector
    Defines task problem for prompt optimization

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
        use_structured_output: if True, the LLM is queried via
            ``model.with_structured_output(..., method="json_schema")``
            using the dedicated pydantic schema; otherwise a plain
            ``invoke()`` is performed and the JSON payload is parsed
            from the raw text response.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        use_structured_output: bool = False,
    ) -> None:
        self.model = model
        self.use_structured_output = use_structured_output

    def _generate(
        self, request: str, schema: BaseModel, field_name: str
    ) -> Any:
        """Generates model output either using structured output from
        langchain (when ``self.use_structured_output`` is True) or a
        plain ``invoke()`` call combined with JSON extraction from text.

        Args:
            request (str): request to LLM
            schema (BaseModel): Pydantic output format (only used when
                structured output is enabled)
            field_name (str): field name to select from output

        Returns:
            Any: generated data
        """
        if self.use_structured_output:
            structured_model = self.model.with_structured_output(
                schema=schema, method="json_schema"
            )
            output = structured_model.invoke(request)
            if isinstance(output, AIMessage):
                output = output.content
            try:
                return getattr(output, field_name)
            except Exception:
                return output[field_name]

        output = self.model.invoke(request)
        if isinstance(output, AIMessage):
            output = output.content
        return extract_json(output)[field_name]

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
        schema = TaskDetectionResponse
        request = TASK_DETECTOR_TEMPLATE.format(query=prompt)

        logger.info("Detecting the task by query")

        task = self._generate(request, schema, "task")

        logger.info(f"Task defined as {task}")

        return task
