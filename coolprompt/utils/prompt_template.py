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
REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.\n"
    "\n"
    "Below are two prompts for {PROBLEM_DESCRIPTION}.\n"
    "You are provided with two prompt versions below, where the second version performs better than the first one.\n"
    "[Worse prompt]\n"
    "{WORSE_PROMPT}\n"
    "[Better prompt]\n"
    "{BETTER_PROMPT}\n"
    "You respond only with one small hint for designing better prompts , based on the two prompt versions and using less than 20 words.\n"
    "I want you to generate only one new hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.\n"
    "Bracket the final hint with <hint> </hint>.\n"
)
REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.\n"
    "\n"
    "Below is your prior long−term reflection on designing prompts for {PROBLEM_DESCRIPTION}.\n"
    "{PRIOR_LONG_TERM_REFLECTION}\n"
    "\n"
    "Below are some newly gained insights.\n"
    "{NEW_SHORT_TERM_REFLECTIONS}\n"
    "\n"
    "Write the constructive hint for designing better prompts, based on prior reflections and new insights and using less than 50 words.\n"
    "I want you to generate only one new constructive hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.\n"
    "Bracket the final hint with <hint> </hint>.\n"
)
REFLECTIVEPROMPT_CROSSOVER_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.\n"
    "Your response outputs prompt text and nothing else.\n"
    "\n"
    "Write a prompt for the task: {PROBLEM_DESCRIPTION}.\n"
    "\n"
    "[Worse prompt]\n"
    "{WORSE_PROMPT}\n"
    "[Better prompt]\n"
    "{BETTER_PROMPT}\n"
    "[Reflection]\n"
    "{SHORT_TERM_REFLECTION}\n"
    "[Improved prompt]\n"
    "Please write an improved prompt, according to the reflection.\n"
    "Bracket the final prompt with <prompt> </prompt>.\n"
)
REFLECTIVEPROMPT_MUTATION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.\n"
    "Your response outputs prompt text and nothing else.\n"
    "\n"
    "Write a prompt for {PROBLEM_DESCRIPTION}.\n"
    "[Prior reflection]\n"
    "{LONG_TERM_REFLECTION}\n"
    "[Prompt]\n"
    "{ELITIST_PROMPT}\n"
    "[Improved prompt]\n"
    "Please write a mutated prompt, according to the reflection.\n"
    "Output prompt only.\n"
    "Bracket the final prompt with <prompt> </prompt>.\n"
)
REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE = (
    "Paraphrase the given prompt text keeping its initial meaning.\n"
    "Prompt: {PROMPT}\n"
    "Create the new variations of this prompt and output them in JSON structure below:\n"
    "{{\n"
    "   \"prompts\": [\n"
    "       \"New prompt 1\",\n"
    "       \"New prompt 2\",\n"
    "       \"New prompt 3\",\n"
    "       ...\n"
    "       \"New prompt {NUM_PROMPTS}\",\n"
    "   ]\n"
    "}}\n"
    "Output JSON data only.\n"
)
REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.\n"
    "Write a prompt that will effectively solve the task: {PROBLEM_DESCRIPTION}.\n"
    "Output prompt only.\n"
    "Bracket the final prompt with <prompt> </prompt>.\n"
)
CLASSIFICATION_TASK_TEMPLATE = "{PROMPT}\n\nAnswer using the label from [{LABELS}].\nGenerate the final answer bracketed with <ans> and </ans>.\n\nInput:\n{INPUT}\n\nResponse:\n"
GENERATION_TASK_TEMPLATE = "{PROMPT}\n\nINPUT:\n{INPUT}\n\nRESPONSE:\n"
CLASSIFICATION_TASK_TEMPLATE = """{PROMPT}

Answer using the label from [{LABELS}].
Generate the final answer bracketed with <ans> and </ans>.

Input:
{INPUT}

Response:
"""
GENERATION_TASK_TEMPLATE = "{PROMPT}\n\nINPUT:\n{INPUT}\n\nRESPONSE:\n"
REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.\n"
    "\n"
    "Below are two prompts for <PROBLEM_DESCRIPTION>.\n"
    "You are provided with two prompt versions below, where the second version performs better than the first one.\n"
    "[Worse prompt]\n"
    "<WORSE_PROMPT>\n"
    "[Better prompt]\n"
    "<BETTER_PROMPT>\n"
    "You respond only with one small hint for designing better prompts , based on the two prompt versions and using less than 20 words.\n"
    "I want you to generate only one new hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.\n"
    "Bracket the final hint with <hint> </hint>."
)
REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.\n"
    "\n"
    "Below is your prior long−term reflection on designing prompts for <PROBLEM_DESCRIPTION>.\n"
    "<PRIOR_LONG_TERM_REFLECTION>\n"
    "\n"
    "Below are some newly gained insights.\n"
    "<NEW_SHORT_TERM_REFLECTIONS>\n"
    "\n"
    "Write the constructive hint for designing better prompts, based on prior reflections and new insights and using less than 50 words.\n"
    "I want you to generate only one new constructive hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.\n"
    "Bracket the final hint with <hint> </hint>.\n"
)
REFLECTIVEPROMPT_CROSSOVER_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.\n"
    "Your response outputs prompt text and nothing else.\n"
    "\n"
    "Write a prompt for the task: <PROBLEM_DESCRIPTION>.\n"
    "\n"
    "[Worse prompt]\n"
    "<WORSE_PROMPT>\n"
    "[Better prompt]\n"
    "<BETTER_PROMPT>\n"
    "[Reflection]\n"
    "<SHORT_TERM_REFLECTION>\n"
    "[Improved prompt]\n"
    "Please write an improved prompt, according to the reflection.\n"
    "Bracket the final prompt with <prompt> </prompt>.\n"
)
REFLECTIVEPROMPT_MUTATION_TEMPLATE = (
    "You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.\n"
    "Your response outputs prompt text and nothing else.\n"
    "\n"
    "Write a prompt for <PROBLEM_DESCRIPTION>.\n"
    "[Prior reflection]\n"
    "<LONG_TERM_REFLECTION>\n"
    "[Prompt]\n"
    "<ELITIST_PROMPT>\n"
    "[Improved prompt]\n"
    "Please write a mutated prompt, according to the reflection.\n"
    "Output prompt only.\n"
    "Bracket the final prompt with <prompt> </prompt>.\n"
)
