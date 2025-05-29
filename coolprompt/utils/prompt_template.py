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
