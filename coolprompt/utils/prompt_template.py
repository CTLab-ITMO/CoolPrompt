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
CLASSIFICATION_TASK_TEMPLATE = """{PROMPT}

Answer using the label from [{LABELS}].
Generate the final answer bracketed with <ans> and </ans>.

Input:
{INPUT}

Response:
"""
GENERATION_TASK_TEMPLATE = "{PROMPT}\n\nINPUT:\n{INPUT}\n\nRESPONSE:\n"
