HYPE_PROMPT_TEMPLATE = """### ROLE ###
You are an expert prompt engineer. Your only task is to generate ONE hypothetical instructive prompt that helps a large language model answer the given query.

### HARD CONSTRAINTS ###
1. LANGUAGE:
   - Output MUST be in the SAME LANGUAGE as the query.
   - NEVER translate or switch languages.
2. CONTENT:
   - Output ONLY the hypothetical instructive prompt — do NOT answer the question.
   - Do NOT include any thinking, planning, or analysis in your output.
3. FORMAT:
   - Wrap the entire prompt in these exact tags: [PROMPT_START] ... [PROMPT_END]
   - No text before [PROMPT_START] or after [PROMPT_END]
4. UNIQUENESS:
   - You MUST return exactly ONE prompt. Never generate more than one.
5. STOP:
   - Your output MUST end with [PROMPT_END] on its own line.
   - Immediately stop after closing [PROMPT_END] tag. Do not continue.

### INPUT ###
Query: <QUERY>

### YOUR OUTPUT FORMAT (strictly one prompt wrapped in tags!) ###
[PROMPT_START]<your hypothetical instructive prompt here>[PROMPT_END]
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below are two prompts for {PROBLEM_DESCRIPTION}.
You are provided with two prompt versions below, where the second version performs better than the first one.
[Worse prompt]
{WORSE_PROMPT}
[Better prompt]
{BETTER_PROMPT}
You respond only with one small hint for designing better prompts , based on the two prompt versions and using less than 20 words.
I want you to generate only one new hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below is your prior long−term reflection on designing prompts for {PROBLEM_DESCRIPTION}.
{PRIOR_LONG_TERM_REFLECTION}

Below are some newly gained insights.
{NEW_SHORT_TERM_REFLECTIONS}

Write the constructive hint for designing better prompts, based on prior reflections and new insights and using less than 50 words.
I want you to generate only one new constructive hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs prompt text and nothing else.

Write a prompt for the task: {PROBLEM_DESCRIPTION}.

[Worse prompt]
{WORSE_PROMPT}
[Better prompt]
{BETTER_PROMPT}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved prompt]
Please write an improved prompt, according to the reflection.
Bracket the final prompt with <prompt> </prompt>.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs prompt text and nothing else.

Write a prompt for {PROBLEM_DESCRIPTION}.
[Prior reflection]
{LONG_TERM_REFLECTION}
[Prompt]
{ELITIST_PROMPT}
[Improved prompt]
Please write a mutated prompt, according to the reflection.
Output prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""

REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE = """Paraphrase the given prompt text keeping its initial meaning.
Prompt: {PROMPT}
Create the new variations of this prompt and output them in JSON structure below:
{{
   "prompts": [
       "New prompt 1",
       "New prompt 2",
       "New prompt 3",
       ...
       "New prompt {NUM_PROMPTS}",
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Write a prompt that will effectively solve the task: {PROBLEM_DESCRIPTION}.
Output prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""

CLASSIFICATION_TASK_TEMPLATE = """{PROMPT}
Answer using the label from [{LABELS}].
Generate the final answer bracketed with <ans> and </ans>.

Input:
{INPUT}

Response:
"""

GENERATION_TASK_TEMPLATE = """{PROMPT}

INPUT:
{INPUT}

RESPONSE:
"""

CLASSIFICATION_TASK_TEMPLATE_HYPE = """{PROMPT}

### HARD CONSTRAINTS ###
1. OUTPUT FORMAT:
   - Output ONLY the final answer in the format: `<ans>LABEL</ans>`
   - LABEL MUST be EXACTLY one item from the list: [{LABELS}]
   - NEVER add explanations, thoughts, or extra text.
   - NEVER modify the label format (e.g., no \"A: Explanation\", ONLY \"A\").
2. STOP CONDITION:
   - IMMEDIATELY stop generating after `</ans>`.

### INPUT ###
{INPUT}

Response:
"""

GENERATION_TASK_TEMPLATE = """{PROMPT}

Generate the final answer bracketed with <ans> and </ans>.

INPUT:
{INPUT}

RESPONSE:
"""
