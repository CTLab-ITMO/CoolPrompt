from pydantic import BaseModel, Field


class TaskDetectionResponse(BaseModel):
    """Response schema for task-type classification.

    Used by :class:`coolprompt.task_detector.detector.TaskDetector`
    to obtain a strict, single-field structured output from the LLM
    when ``use_structured_output=True``.
    """

    task: str = Field(description="The name of the detected task.")
