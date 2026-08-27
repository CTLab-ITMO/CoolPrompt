REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_FIXED_ROLE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below are two prompt configurations for {PROBLEM_DESCRIPTION}.
You are provided with two prompt versions below, where the second version performs better than the first one.

The System Role is FIXED and is the same for both prompts:
Role: {BETTER_PROMPT_ROLE}

[Worse prompt text]
Prompt: {WORSE_PROMPT_TEXT}
[Better prompt text]
Prompt: {BETTER_PROMPT_TEXT}

You respond only with one small hint for designing better prompts TEXT, based on the two prompt versions and fixed role, using less than 20 words.
I want you to generate only one new hint for the prompt text itself. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_FIXED_ROLE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below is your prior long−term reflection on designing prompts for {PROBLEM_DESCRIPTION}.
{PRIOR_LONG_TERM_REFLECTION}

Below are some newly gained insights.
{NEW_SHORT_TERM_REFLECTIONS}

Write the constructive hint for designing better prompt TEXTS, based on prior reflections and new insights and using less than 50 words.
The System Role is FIXED, so focus only on optimizing the user prompt text.
I want you to generate only one new constructive hint. For example, you can try to recommend word replacements, active/positive voice conversions, adding words or delete words.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_FIXED_ROLE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs a JSON object with one field: "prompt".

The System Role is FIXED and cannot be changed:
Role: {BETTER_PROMPT_ROLE}

Write a new prompt text for the task: {PROBLEM_DESCRIPTION}.

[Worse prompt text]
Prompt: {WORSE_PROMPT_TEXT}
[Better prompt text]
Prompt: {BETTER_PROMPT_TEXT}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved prompt configuration]
Please write an improved prompt text, according to the reflection, optimized for the fixed role above.
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_FIXED_ROLE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs a JSON object with one field: "prompt".

The System Role is FIXED and cannot be changed:
Role: {ELITIST_PROMPT_ROLE}

Write a mutated prompt text for {PROBLEM_DESCRIPTION}.
[Prior reflection]
{LONG_TERM_REFLECTION}
[Current elitist prompt text]
Prompt: {ELITIST_PROMPT_TEXT}
[Mutated prompt configuration]
Please write a mutated prompt text, according to the reflection, optimized for the fixed role above.
Output JSON data only.
"""

REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_FIXED_ROLE = """Paraphrase the given prompt text keeping its initial meaning. The role is fixed.

Fixed Role: {ROLE}
Prompt: {PROMPT}

Create {NUM_PROMPTS} new variations of this prompt text (optimized for the fixed role) and output them in JSON structure below:
{{
   "prompts": [
       "New prompt 1",
       "New prompt 2",
       ...
       "New prompt {NUM_PROMPTS}"
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_FIXED_ROLE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Write a prompt text that will effectively solve the task: {PROBLEM_DESCRIPTION}.
The System Role is FIXED and provided separately. Focus only on the prompt text.
Output a JSON object with one field: "prompt".
"""
