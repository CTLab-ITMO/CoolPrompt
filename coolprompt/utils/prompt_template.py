NAIVE_PROMPT_TEMPLATE = (
    "You will be given with user's prompt, rewrite it to maximize its effectiveness for LLMs.\n"
    "Apply transformations: structure, specifics, remove ambiguity, add example, keep intent.\n"
    "Only output the rewritten prompt, with no explanation or formatting.\n"
    "Start rewritten prompt with <START> tag and end with <END> tag, don't write anything after <END> tag.\n"
    "\n"
    "Prompt you have to rewrite:\n"
    "<PROMPT>\n"
    "Rewritten prompt:\n"
)
CLASSIFICATION_TASK_TEMPLATE = "<PROMPT>\n\nAnswer using the label from [<LABELS>].\nGenerate the final answer bracketed with <ans> and </ans>.\n\nThe input:\n<INPUT>\n\nResponse:\n"
GENERATION_TASK_TEMPLATE = "<PROMPT>\n\nINPUT:\n<INPUT>\n\nRESPONSE:\n"

