REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_TEXT_ONLY = """Create {NUM_PROMPTS} concise variations of the task instruction below.

Task instruction: {PROMPT}

Rules:
- Each variation must be 1-2 sentences only.
- Preserve the output format requirement exactly (e.g. numeric label, single word, etc.).
- Change wording, emphasis, or directness — do not add extra sentences or explanations.
- Do not include any system role or behavior description in the instruction.

Output JSON:
{{
   "prompts": [
       "Variation 1",
       "Variation 2",
       ...
       "Variation {NUM_PROMPTS}"
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_TEXT_ONLY = """You are an expert in prompt optimization. Give a brief hint for writing better task instructions.

Task: {PROBLEM_DESCRIPTION}

Two task instructions were tested.
[Worse instruction] (score: {WORSE_SCORE})
{WORSE_PROMPT_TEXT}
[Better instruction] (score: {BETTER_SCORE})
{BETTER_PROMPT_TEXT}

Why does the better instruction lead to higher scores? Focus on wording, directness, or clarity differences.
Give one actionable hint in less than 20 words.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_TEXT_ONLY = """You are an expert in prompt optimization. Synthesize hints for writing better task instructions.

Task: {PROBLEM_DESCRIPTION}

Best-performing task instructions found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Study these top instructions carefully: what phrasing, verb choice, or framing distinguishes the higher-scoring ones? What do they have in common that lower-scoring instructions lack?

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Write one updated actionable hint (less than 40 words) about what makes a task instruction score higher on this specific task.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_TEXT_ONLY = """You are an expert in prompt optimization. Combine two task instructions into a better one.

Task: {PROBLEM_DESCRIPTION}

[Worse instruction]
{WORSE_PROMPT_TEXT}
[Better instruction]
{BETTER_PROMPT_TEXT}
[Reflection]
{SHORT_TERM_REFLECTION}

Write a new, improved task instruction.
Rules:
- 1-2 sentences only. No role or behavior description.
- Keep the output format requirement (how the answer must be returned) from the better instruction.
- No motivational or filler language.
Bracket the final instruction with <prompt> </prompt>.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_TEXT_ONLY = """You are an expert in prompt optimization. Improve the task instruction below.

Task: {PROBLEM_DESCRIPTION}

[Accumulated insight on what works for this task]
{LONG_TERM_REFLECTION}

[Current best task instruction] (score: {ELITIST_SCORE})
{ELITIST_PROMPT_TEXT}

[Examples where the current instruction most often fails]
Each example shows the input, the wrong answer the model gave, and the correct answer.
{BAD_EXAMPLES}

Analyze these failure cases before writing:
- What type of inputs does the model get wrong?
- Is there a common pattern (e.g., ambiguous phrasing, specific input type, edge case)?
- What does the correct answer reveal about what the instruction fails to convey?

Write a mutated instruction that directly addresses the identified failure pattern.
Rules:
- 1-2 sentences only. No role or behavior description.
- Keep the output format requirement intact (e.g. numeric label, exact phrasing for the answer format).
- Do not add motivational phrases, emotional language, or explanations targeted at the reader.
- Must be meaningfully different from the current instruction.
Bracket the final instruction with <prompt> </prompt>.
"""

REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_TEXT_ONLY = """Write a concise task instruction for the following task.

Task: {PROBLEM_DESCRIPTION}

Rules:
- 1-2 sentences only. No role or behavior description.
- Include how the answer should be returned (output format).
- Be direct and clear.
Bracket the final instruction with <prompt> </prompt>.
"""
