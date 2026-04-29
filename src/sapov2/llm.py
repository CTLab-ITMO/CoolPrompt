"""LLM client helpers for SAPO v2."""

from __future__ import annotations

from langchain_openrouter import ChatOpenRouter

from .schema import CandidateGenerationResponse, ReasoningTemplate, ResponseTemplate

_RESPONSE_SCHEMAS = {
    "base": ResponseTemplate,
    "reasoning": ReasoningTemplate,
    "candidate_generation": CandidateGenerationResponse,
}


def get_openrouter_llm(
    model_name: str = "anthropic/claude-sonnet-4.5",
    response_type: str = "base",
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """Create a structured OpenRouter LLM client."""
    schema = _RESPONSE_SCHEMAS.get(response_type)
    if schema is None:
        supported = ", ".join(sorted(_RESPONSE_SCHEMAS))
        raise ValueError(f"Unsupported response_type={response_type!r}. Supported: {supported}")

    model = ChatOpenRouter(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
    )
    return model.with_structured_output(schema, method="json_schema")
