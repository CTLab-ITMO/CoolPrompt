from pydantic import BaseModel, Field


class DeepEvalJudgeResponse(BaseModel):
    """Fallback response schema for the DeepEval LangChain wrapper.

    DeepEval's ``DeepEvalBaseLLM.generate``/``a_generate`` contract may
    pass an explicit pydantic ``schema`` describing the expected JSON
    payload (used by GEval/DAG judges). When the caller does not provide
    one but structured output is still enabled on the wrapper, this
    schema is used as a safe default: it asks the model to return its
    answer as a single plain-text ``response`` field, which can be
    consumed exactly like the legacy unstructured string output.
    """

    response: str = Field(
        description=(
            "The model's full textual answer to the prompt. Contains "
            "the answer only — no preamble, no markdown fences, no XML "
            "tags, no trailing meta-comments. If the prompt asks for a "
            "specific format (JSON, number, list, etc.) follow it "
            "strictly inside this field."
        )
    )
