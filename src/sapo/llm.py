"""Utilities for creating SAPO-compatible OpenRouter LLM clients."""

from __future__ import annotations

from langchain_openrouter import ChatOpenRouter

from .schema import ReasoningCandGenTemplate, ReasoningTemplate, ResponseTemplate

_RESPONSE_SCHEMAS = {
    "base": ResponseTemplate,
    "reasoning": ReasoningTemplate,
    "cand_gen": ReasoningCandGenTemplate,
}


def get_openrouter_llm(
    model_name: str = "anthropic/claude-sonnet-4.5",
    response_type: str = "base",
    temperature: float = 0.0,
):
    """Create a ChatOpenRouter client wrapped with a structured response schema.

    Args:
        model_name: OpenRouter model id.
        response_type: One of ``base``, ``reasoning``, or ``cand_gen``.
        temperature: Sampling temperature for generation.

    Returns:
        A LangChain model instance configured with JSON schema output.

    Raises:
        ValueError: If an unknown ``response_type`` is passed.
    """
    schema = _RESPONSE_SCHEMAS.get(response_type)
    if schema is None:
        supported_types = ", ".join(sorted(_RESPONSE_SCHEMAS))
        raise ValueError(
            f"Unsupported response_type: {response_type!r}. "
            f"Supported values: {supported_types}."
        )

    model = ChatOpenRouter(
        model=model_name,
        temperature=temperature,
        max_tokens=1024,
        max_retries=3,
    )
    return model.with_structured_output(schema, method="json_schema")


def get_final_prompt(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Build a two-message chat payload from system and user prompts."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
