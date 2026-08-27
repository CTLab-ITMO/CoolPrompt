REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_BASE = """You are an expert in prompt optimization. Your task is to give hints to design better prompt configurations.

Below are two prompt configurations for {PROBLEM_DESCRIPTION}.
Each configuration has two components:
- system_behavior: behavioral instructions defining HOW the AI should reason, verify, and process information (NOT a persona or job title)
- task_description: the specific task instruction defining WHAT the AI should do

The second configuration performs better than the first one.
[Worse configuration]
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
[Better configuration]
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
Analyze differences in both system_behavior and task_description separately.
Consider WHY the better configuration works better. Focus on actionable changes.
Respond with one concise hint (less than 30 words) covering what to change in system_behavior and what to change in task_description.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_BASE = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with two fields: "system_behavior" and "task_description".

Write a new prompt configuration for the task: {PROBLEM_DESCRIPTION}.

[Worse configuration]
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
[Better configuration]
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved configuration]
Combine the strongest aspects of both configurations according to the reflection.
You may take the system_behavior approach from one configuration and the task_description approach from the other.

Rules:
- "system_behavior" MUST describe specific ACTIONS and REASONING STEPS — not identity or expertise.
  Do NOT write: "You are an expert in X", "A specialist in Y", "Domain professional"
  DO write: "Before answering, verify X. Check for Y. If Z, then..."
- "system_behavior" MUST be a complete instruction of at least 8 words.
- "task_description" MUST be a clear, actionable task instruction.
- system_behavior and task_description must complement each other without duplicating instructions.
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_BASE = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with two fields: "system_behavior" and "task_description".

Write a mutated prompt configuration for {PROBLEM_DESCRIPTION}.
[Prior reflection]
{LONG_TERM_REFLECTION}
[Current elitist configuration]
System Behavior: {ELITIST_PROMPT_ROLE}
Task Description: {ELITIST_PROMPT_TEXT}
[Mutated configuration]
IMPORTANT for system_behavior: Aggressively reimagine it. Create a fundamentally different behavioral specification.
Do NOT describe identity (who you are). Describe BEHAVIOR (what to do, what to check, how to reason).
system_behavior MUST be a complete instruction of at least 8 words — NOT a persona or job title.
Bad examples: "Topic Analyst", "Data Specialist", "You are an expert in X"
Good examples: "Before responding, verify the key facts in the input. Check for consistency and relevance.", "Identify the core information first, then formulate a concise and accurate response."
The task_description may be changed moderately, applying the accumulated reflection.
system_behavior and task_description must not repeat the same instructions.
Output JSON data only.
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO ="""You are an expert in prompt optimization. Your task is to give hints to design better prompt configurations.

Below are two prompt configurations for {PROBLEM_DESCRIPTION}.
Each configuration has two components:
- task_description: WHAT the model should do and how to format the answer. Covers the task itself and output requirements.
- system_behavior: HOW the model should approach the task — reasoning strategy, key things to check, special considerations. Can start with a brief role ("You are X") only if immediately followed by a concrete behavioral instruction.

The second configuration performs better than the first one.
[Worse configuration] (score: {WORSE_SCORE})
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
[Better configuration] (score: {BETTER_SCORE})
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
Analyze differences in both components separately.
Consider WHY the better configuration scored higher on this specific task.
Respond with one concise hint (less than 30 words) about what makes the better configuration work.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO = """You are an expert in prompt optimization. Your task is to give hints to design better prompt configurations.

Task: {PROBLEM_DESCRIPTION}

Best-performing configurations found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Study these top configurations: what patterns in system_behavior and task_description do the higher-scoring ones share?

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Write one updated actionable hint (less than 50 words) about what makes a configuration score highest on this specific task.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with two fields: "system_behavior" and "task_description".

Write a new prompt configuration for the task: {PROBLEM_DESCRIPTION}.

[Worse configuration] (score: {WORSE_SCORE})
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
[Better configuration] (score: {BETTER_SCORE})
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved configuration]
Combine the strongest aspects of both configurations according to the reflection.
Your goal is to score HIGHER than {BETTER_SCORE}.

Field rules:
- "task_description": WHAT to do and how to format the answer. Clear and actionable. 1-2 sentences.
- "system_behavior": HOW to approach the task — reasoning strategy, what to prioritize, specific checks.
  You CAN start with "You are [brief role]" if you immediately follow it with a concrete behavioral instruction.
  Example: "You are a careful analyst. Focus on the most relevant detail and verify it matches the expected format."
  NOT enough: "You are an expert." — must say what to DO or CHECK.
  1-2 sentences, 8-25 words.
- The two fields must cover different aspects — task_description covers the task itself, system_behavior covers the approach. Do not repeat the same instruction in both.
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with two fields: "system_behavior" and "task_description".

Task: {PROBLEM_DESCRIPTION}
[Accumulated insight on what works for this task]
{LONG_TERM_REFLECTION}

[Current best configuration] (score: {ELITIST_SCORE})
System Behavior: {ELITIST_PROMPT_ROLE}
Task Description: {ELITIST_PROMPT_TEXT}

[Examples where the current configuration most often fails]
Each example shows the input, the wrong answer the model gave, and the correct answer.
{BAD_EXAMPLES}

Analyze these failure cases before writing:
- Does the failure come from the task instruction (task_description) or from the behavioral approach (system_behavior)?
- What specific reasoning step or focus is missing that would fix these cases?
- What change to either field would prevent these specific errors?

Write a mutated configuration that directly addresses the identified failure pattern and aims to score above {ELITIST_SCORE}.

Field rules:
- "task_description": WHAT to do and how to format the answer. 1-2 sentences.
- "system_behavior": HOW to approach the task — reasoning strategy, what to check, special considerations.
  You CAN start with "You are [brief role]" if you immediately follow it with a concrete behavioral instruction.
  Example: "You are a careful analyst. Focus on the most relevant detail before committing to an answer."
  NOT enough: "You are an expert." — must say what to DO or CHECK.
  1-2 sentences, 8-25 words.
- The two fields must cover different aspects. Do not repeat the same instruction in both.
Output JSON data only.
"""

REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO = """Create diverse variations of the given prompt configuration.
System Behavior: {ROLE}
Task Description: {PROMPT}

Create {NUM_PROMPTS} variations with maximum diversity.
Each variation must have a meaningfully DIFFERENT system_behavior that takes a unique behavioral approach.

Field rules:
- "task_description": WHAT to do and how to format the answer. Vary wording while preserving task intent.
- "system_behavior": HOW to approach the task — reasoning strategy, what to prioritize, specific checks. 1-2 sentences.
  You CAN start with "You are [brief role]" if you immediately follow it with a concrete behavioral instruction.
  Examples: "You are a precise reader. Focus on the key entity before formulating a response.",
            "Before responding, identify the main requirement and verify your answer matches it.",
            "You are a methodical solver. Break the input into parts and handle each systematically."
  NOT enough: "You are an expert." — must state what to DO or CHECK.
- The two fields must cover different aspects. Do not repeat the same instruction in both.

Output them in JSON structure below:
{{
   "prompts": [
       {{"system_behavior": "behavior 1", "task_description": "task 1"}},
       {{"system_behavior": "behavior 2", "task_description": "task 2"}},
       ...
       {{"system_behavior": "behavior {NUM_PROMPTS}", "task_description": "task {NUM_PROMPTS}"}},
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO = """Generate a simple initial prompt configuration for the task: {PROBLEM_DESCRIPTION}.

Output a JSON object with exactly two fields:
- "system_behavior": A SHORT behavioral hint (1 sentence, max 15 words) about HOW to approach the task.
  Write a simple practical instruction, NOT a role or identity.
  Bad: "You are an expert in X", "Data Analyst"
  Good: "Think step by step before answering.", "Check your reasoning carefully."
- "task_description": A SHORT task instruction (1 sentence, max 15 words) about WHAT to do.

IMPORTANT: Keep BOTH fields brief and generic. These are starting points that will be refined later.
Do NOT write elaborate multi-step strategies. Do NOT include specific examples or edge cases.
Output JSON only.
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_3F = """You are an expert in prompt optimization. Your task is to give hints to design better prompt configurations.

Below are two prompt configurations for {PROBLEM_DESCRIPTION}.
Each configuration has three components:
- task_description: WHAT the model should do and how to format the answer. Covers the task itself and output requirements.
- system_behavior: HOW the model should approach the task — reasoning strategy, what to check, special considerations. Can start with a brief role ("You are X") only if immediately followed by a concrete behavioral instruction.
- output_constraints: FORMAT and STYLE rules only — length limits, structure, tone, what to omit. Examples: "One sentence only.", "No extra context.", "Use subject-verb-object order."

The second configuration performs better than the first one.
[Worse configuration] (score: {WORSE_SCORE})
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
Output Constraints: {WORSE_PROMPT_CONSTRAINTS}
[Better configuration] (score: {BETTER_SCORE})
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
Output Constraints: {BETTER_PROMPT_CONSTRAINTS}
Analyze differences in all three components separately.
Consider WHY the better configuration scored higher on this specific task.
Respond with one concise hint (less than 30 words) about what makes the better configuration work.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO_3F = """You are an expert in prompt optimization. Your task is to give hints to design better prompt configurations.

Task: {PROBLEM_DESCRIPTION}

Best-performing configurations found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Study these top configurations: what patterns in system_behavior, task_description, and output_constraints do the higher-scoring ones share?

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Write one updated actionable hint (less than 50 words) about what makes a configuration score highest on this task.
Cover three aspects: what system_behavior patterns work best, what task_description patterns work best, and what output_constraints are most effective.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_3F = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with three fields: "system_behavior", "task_description", and "output_constraints".

Write a new prompt configuration for the task: {PROBLEM_DESCRIPTION}.

[Worse configuration] (score: {WORSE_SCORE})
System Behavior: {WORSE_PROMPT_ROLE}
Task Description: {WORSE_PROMPT_TEXT}
Output Constraints: {WORSE_PROMPT_CONSTRAINTS}
[Better configuration] (score: {BETTER_SCORE})
System Behavior: {BETTER_PROMPT_ROLE}
Task Description: {BETTER_PROMPT_TEXT}
Output Constraints: {BETTER_PROMPT_CONSTRAINTS}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved configuration]
Combine the strongest aspects of both configurations according to the reflection.
Your goal is to score HIGHER than {BETTER_SCORE}.

Field rules:
- "task_description": WHAT to do and how to format the answer. 1-2 sentences. Clear and actionable.
- "system_behavior": HOW to approach the task — reasoning strategy, what to prioritize, specific checks.
  You CAN start with "You are [brief role]" if immediately followed by a concrete behavioral instruction.
  Example: "You are a careful analyst. Focus on the most relevant detail and verify it matches the expected format."
  NOT enough: "You are an expert." — must say what to DO or CHECK. 1-2 sentences, 8-25 words.
- "output_constraints": FORMAT and STYLE rules only — length, structure, what to omit. 1-2 short rules.
  Example: "One sentence only. No extra context beyond the main point."
- The three fields must cover different aspects — no repeated instructions across fields.
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_3F = """You are an expert in prompt optimization. Your task is to design prompt configurations that effectively solve tasks.
Your response outputs a JSON object with three fields: "system_behavior", "task_description", and "output_constraints".

Task: {PROBLEM_DESCRIPTION}
[Accumulated insight on what works for this task]
{LONG_TERM_REFLECTION}

[Current best configuration] (score: {ELITIST_SCORE})
System Behavior: {ELITIST_PROMPT_ROLE}
Task Description: {ELITIST_PROMPT_TEXT}
Output Constraints: {ELITIST_PROMPT_CONSTRAINTS}

[Examples where the current configuration most often fails]
Each example shows the input, the wrong answer the model gave, and the correct answer.
{BAD_EXAMPLES}

Analyze these failure cases before writing:
- Does the failure come from task_description (wrong instruction), system_behavior (wrong approach), or output_constraints (wrong format rule)?
- What specific change to which field would prevent these errors?

Write a mutated configuration that directly addresses the identified failure pattern and aims to score above {ELITIST_SCORE}.

Field rules:
- "task_description": WHAT to do and how to format the answer. 1-2 sentences.
- "system_behavior": HOW to approach the task — reasoning strategy, what to check, special considerations.
  You CAN start with "You are [brief role]" if immediately followed by a concrete behavioral instruction.
  Example: "You are a careful analyst. Focus on the most relevant detail before committing to an answer."
  NOT enough: "You are an expert." — must say what to DO or CHECK. 1-2 sentences, 8-25 words.
- "output_constraints": FORMAT and STYLE rules only — length limits, structure, what to include or omit.
  Example: "One sentence only. No additional context. Start directly with the answer."
- The three fields must cover different aspects. No repeated instructions across fields.
Output JSON data only.
"""

REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO_3F = """Create diverse variations of the given prompt configuration.
System Behavior: {ROLE}
Task Description: {PROMPT}
Output Constraints: {CONSTRAINTS}
Create {NUM_PROMPTS} variations with maximum diversity.
Each variation must have meaningfully DIFFERENT system_behavior, task_description, and output_constraints.
- "system_behavior": Start with a role identity, then describe reasoning steps.
- "task_description": Vary wording while preserving task intent.
- "output_constraints": Vary format, length, tone, or quality rules.
Output them in JSON structure below:
{{
   "prompts": [
       {{"system_behavior": "New behavior 1", "task_description": "New task 1", "output_constraints": "New constraints 1"}},
       {{"system_behavior": "New behavior 2", "task_description": "New task 2", "output_constraints": "New constraints 2"}},
       ...
       {{"system_behavior": "New behavior {NUM_PROMPTS}", "task_description": "New task {NUM_PROMPTS}", "output_constraints": "New constraints {NUM_PROMPTS}"}},
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_3F = """Generate a simple initial prompt configuration for the task: {PROBLEM_DESCRIPTION}.

Output a JSON object with exactly three fields:
- "system_behavior": A brief role identity + practical reasoning hint (max 20 words).
  Example: "You are a logical analyst. Think step by step before answering."
- "task_description": A SHORT task instruction (1 sentence, max 15 words).
- "output_constraints": Brief rules about output format or style (max 15 words).
  Example: "Be concise. Follow the requested format strictly."

IMPORTANT: Keep ALL fields brief and generic. These are starting points that will be refined later.
Output JSON only.
"""
