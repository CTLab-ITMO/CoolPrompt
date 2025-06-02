NAIVE_PROMPT_TEMPLATE = (
    """You will be given with user's prompt, rewrite it to maximize its effectiveness for LLMs.
    Apply transformations: structure, specifics, remove ambiguity, add example, keep intent.
    Only output the rewritten prompt, with no explanation or formatting.
    Start rewritten prompt with <START> tag and end with <END> tag, don't write anything after <END> tag.
    Prompt you have to rewrite:
    <PROMPT>
    Rewritten prompt:
    """
)
