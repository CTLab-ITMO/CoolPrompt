from pydantic import BaseModel, Field


class ClassificationAnswerResponse(BaseModel):
    """Response schema for classification-task answers produced by the
    target model during evaluation.

    The model MUST pick exactly one label from the list provided in the
    prompt. The value is matched against the dataset's label space later
    (case-sensitive, see ``ClassificationMetric._encode_labels``), so it
    must reproduce the label string verbatim.
    """

    answer: str = Field(
        description=(
            "The single chosen label for the given input. MUST be one of "
            "the labels enumerated in the prompt's `[LABELS]` list, "
            "reproduced EXACTLY as given (same casing, same spelling, no "
            "surrounding quotes, no extra punctuation, no explanation). "
            "Do not output multiple labels, do not add commentary."
        )
    )


class GenerationAnswerResponse(BaseModel):
    """Response schema for free-form generation-task answers produced by
    the target model during evaluation.
    """

    answer: str = Field(
        description=(
            "The final answer to the task as plain text. Contains the "
            "answer only — no preamble like 'Sure, here is...', no "
            "trailing meta-comments, no markdown fences, no XML tags. "
            "If the task requires a specific format (number, code "
            "snippet, list, etc.) follow it strictly inside this field."
        )
    )


class JudgeScoreResponse(BaseModel):
    """Response schema for the LLM-as-a-judge metric.

    The judge rates a candidate response on a single criterion using an
    integer scale ``1..metric_ceil`` (``metric_ceil`` is provided by the
    caller, default 10). Only the numeric score is required — no
    free-form justification.
    """

    score: int = Field(
        ge=1,
        description=(
            "Integer score for the requested criterion. MUST be between "
            "1 (worst) and the maximum value `metric_ceil` provided in "
            "the prompt (inclusive). Do not output a fractional value, "
            "do not output a range, do not add any explanation — the "
            "single integer score is the entire answer."
        ),
    )
