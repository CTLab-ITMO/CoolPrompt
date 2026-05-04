from pydantic import BaseModel, Field


class OptimizedPromptResponse(BaseModel):
    """Response schema for HyPE-generated optimized prompt."""

    result_prompt: str = Field(
        description=(
            "A single hypothetical instructive prompt that, when given to "
            "another LLM, would lead it to correctly solve the same "
            "underlying task as the original user query. "
            "HARD CONSTRAINTS: "
            "(1) Do NOT directly answer or solve the original query — output "
            "an instructional prompt only, not the answer; "
            "(2) Do NOT include chain-of-thought, reasoning steps, "
            "explanations, or meta-commentary about how the prompt was "
            "produced; "
            "(3) Preserve the original language of the user query "
            "(answer in the same language); "
            "(4) Preserve any code blocks, placeholders, variables and "
            "special formatting present in the original query verbatim; "
            "(5) Stay strictly on the same topic/domain as the original "
            "query — do not drift to unrelated tasks even if the query is "
            "underspecified; "
            "(6) Return the prompt content only — no [PROMPT_START]/"
            "[PROMPT_END] or other wrapping tags, no surrounding quotes, "
            "no extra commentary."
        )
    )
