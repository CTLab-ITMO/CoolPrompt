from pydantic import BaseModel, Field


class ParaphrasedVariantResponse(BaseModel):
    """Response schema for HyPER paraphrasing of the current best prompt."""

    paraphrased_prompt: str = Field(
        description=(
            "An alternative version (paraphrase) of the original prompt that: "
            "(1) preserves the original meaning, all key details, and "
            "the original language verbatim; "
            "(2) uses noticeably different words, sentence structure, and/or "
            "tone (e.g., more formal, more casual, or more creative) "
            "compared to the original; "
            "(3) varies in length by at most ±10% relative to the original "
            "prompt; "
            "(4) preserves any code blocks, placeholders and special "
            "formatting present in the original. "
            "Output the paraphrased prompt content only — plain text, "
            "no XML tags, no quotes, no surrounding commentary."
        )
    )
