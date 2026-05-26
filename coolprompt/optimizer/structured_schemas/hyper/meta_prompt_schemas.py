from pydantic import BaseModel, Field


class ResultPromptResponse(BaseModel):
    """Response schema for HyPER meta-prompt single-step optimization."""

    result_prompt: str = Field(
        description=(
            "A single optimized prompt that follows the section structure "
            "requested by the meta-prompt (typically: Role, Task context, "
            "Instructions, Output requirements) and applies all listed "
            "recommendations and hard constraints. "
            "HARD CONSTRAINTS: "
            "(1) Do NOT answer the original user query — produce an "
            "instructional prompt only; "
            "(2) Do NOT include chain-of-thought, reasoning steps, "
            "explanations, or meta-commentary about how the prompt was "
            "produced; "
            "(3) Preserve the language of the user's query verbatim; "
            "(4) Preserve any code blocks, inline code, placeholders, "
            "identifiers, numerical values and special formatting present "
            "in the user query; "
            "(5) Stay strictly on the same task/domain as the user query — "
            "do not drift to unrelated tasks even if the query is "
            "underspecified; "
            "(6) Return the prompt content only — NO <result_prompt> tags, "
            "NO surrounding quotes, NO leading or trailing labels such as "
            "'Result prompt:'; "
            "(7) The prompt body itself MUST be valid Markdown: use "
            "headings (e.g. '#', '##') for each section, bulleted lists "
            "for enumerations and fenced code blocks for any code or "
            "pseudo-code."
        )
    )


class ParaphrasedVariantResponse(BaseModel):
    """Response schema for paraphrasing the current best HyPER prompt."""

    paraphrased_prompt: str = Field(
        description=(
            "An alternative version (paraphrase) of the original structured "
            "prompt that: "
            "(1) PRESERVES THE SECTION STRUCTURE EXACTLY — same section "
            "headings (e.g. '# Role', '# Task context', '# Instructions', "
            "'# Output requirements', or whatever headings the original "
            "uses) in the SAME ORDER, no new sections added, no existing "
            "sections removed, no headings renamed; "
            "(2) within each section, REWRITES THE BODY SUBSTANTIALLY — "
            "different words, different sentence structure, different "
            "voice (active/passive), reordering ideas where natural; aim "
            "for noticeable rewording rather than swapping a few synonyms; "
            "(3) preserves the core MEANING and INTENT of every section, "
            "the original LANGUAGE, and any code, inline code, "
            "identifiers and numerical values verbatim; "
            "(4) keeps per-section length within ±20% of the original; "
            "(5) if the original has no clear section headings, "
            "paraphrases the whole text following the same rewording "
            "standards. "
            "Output the paraphrased prompt content only — plain text, "
            "no XML tags, no quotes, no leading or trailing labels such "
            "as 'Alternative prompt:'."
        )
    )
