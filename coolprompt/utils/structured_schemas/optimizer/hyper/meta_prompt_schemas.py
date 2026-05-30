from pydantic import BaseModel, Field


class ResultPromptResponse(BaseModel):
    """Response schema for HyPER meta-prompt single-step optimization.

    Note: behavioral rules (language preservation, code preservation,
    markdown formatting, do-not-answer-the-query, section structure, etc.)
    are enforced by the meta-prompt body itself (see
    ``coolprompt.utils.prompt_templates.hyper_templates``). The schema
    only defines field semantics, so as not to duplicate or contradict
    those instructions in the structured-output channel.
    """

    result_prompt: str = Field(
        description=(
            "The optimized instructional prompt produced from the "
            "meta-prompt. Return the prompt content only — without "
            "<result_prompt> XML tags, without surrounding quotes, and "
            "without leading or trailing labels such as 'Result prompt:'. "
            "All other formatting, structural and content rules are "
            "specified by the meta-prompt body."
        )
    )


class ParaphrasedVariantResponse(BaseModel):
    """Response schema for paraphrasing the current best HyPER prompt.

    Note: the full paraphrase contract (rewording depth, language and
    code preservation, optional section-structure preservation when
    headings are present) is specified by ``PARAPHRASE_PROMPT`` in
    ``coolprompt.utils.prompt_templates.hyper_templates``. The schema
    only describes the field semantics.
    """

    paraphrased_prompt: str = Field(
        description=(
            "A paraphrased variant of the input prompt that preserves "
            "its core meaning and intent, its original language, and "
            "any code, inline code, identifiers and numerical values "
            "verbatim. Return the paraphrased prompt content only — "
            "plain text, no XML tags, no surrounding quotes, no "
            "leading or trailing labels such as 'Alternative prompt:'."
        )
    )
