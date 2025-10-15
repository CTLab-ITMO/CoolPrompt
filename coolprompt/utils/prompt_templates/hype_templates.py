HYPE_PROMPT_TEMPLATE = (
    "You are an expert prompt engineer. Your only task is to "
    "generate a hypothetical instructive prompt that would help "
    "a large language model effectively answer the following query. "
    "The prompt must solve the same underlying task as the original query while being more effective.\n"
    "### HARD CONSTRAINTS ###\n"
    "1. LANGUAGE:\n"
    "   - Output MUST be in the EXACT SAME LANGUAGE as the query.\n"
    "2. CONTENT:\n"
    "   - Output ONLY the hypothetical instructive prompt - do NOT answer the original query directly.\n"
    "   - The hypothetical prompt must solve the same task as the original query provided by user.\n"
    "   - If the original query contains any code snippets, you must include it in final prompt.\n"
    "3. TECHNICAL PRESERVATION:\n"
    "   - Code blocks must be preserved with original syntax and formatting.\n"
    "   - Variables, placeholders ({{var}}), and technical terms kept unchanged.\n"
    "   - Markdown and special formatting replicated precisely.\n"
    "### YOUR OUTPUT FORMAT ###\n"
    "[PROMPT_START]<your hypothetical instructive prompt here>[PROMPT_END]\n"
    "### INPUT ###\n"
    "User's query: {QUERY}\n"
    "Problem description: {PROBLEM_DESCRIPTION}\n"
    "### OUTPUT ###\n"
    "Hypothetical Instructive Prompt: "
)

CLASSIFICATION_TASK_TEMPLATE_HYPE = """{PROMPT}

Answer using exactly one label from [{LABELS}].
Generate the final answer bracketed with <ans> and </ans>.
Examples:
1. Labels are [(A), (B), (C)] and you chose the first option
       Output will be: <ans>(A)</ans>
2. Labels are [A, B, C] and you chose the first option
       Output will be: <ans>A</ans>

Input:
{INPUT}

Response:
"""

GENERATION_TASK_TEMPLATE_HYPE = """{PROMPT}

Provide a direct answer without additional explanations or commentary.
Generate the final answer bracketed with <ans> and </ans>.

INPUT:
{INPUT}

RESPONSE:
"""
