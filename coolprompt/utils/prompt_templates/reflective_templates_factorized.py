
REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_ROLE_ONLY = """Create {NUM_PROMPTS} diverse system_behavior variants for a fixed task instruction.

Task: {PROBLEM_DESCRIPTION}
Task instruction (frozen — do not change): {PROMPT}

Current system_behavior:
{ROLE}

Rules:
- Each system_behavior must be 1 sentence, between 8 and 25 words.
- You CAN start with "You are X" if you immediately follow it with a concrete behavior or focus.
  It is NOT enough to only name a role — you must say what the model should DO or NOTICE.
- Vary the framing: some variants should name a focus area, some a cognitive strategy, some a domain angle.
- The variants should differ meaningfully from each other.
- Do NOT copy examples from below — they are for illustration only, from unrelated domains.
  Example (translation task): "You are a careful translator. Preserve the original register and avoid literal word-for-word rendering."
  Example (legal task): "Identify the main obligation being described before formulating an answer."
  Example (coding task): "You are a code reviewer. Flag potential edge cases as well as the obvious issue."

Output JSON:
{{
   "prompts": [
       {{"system_behavior": "variant 1"}},
       {{"system_behavior": "variant 2"}},
       ...
       {{"system_behavior": "variant {NUM_PROMPTS}"}}
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_ROLE_ONLY = """You are an expert in prompt optimization. Give a hint for writing better system_behavior instructions.

Task: {PROBLEM_DESCRIPTION}
Task instruction (fixed): {FROZEN_PROMPT_TEXT}

Two system_behavior instructions were tested.
[Worse system_behavior] (score: {WORSE_SCORE})
{WORSE_PROMPT_ROLE}
[Better system_behavior] (score: {BETTER_SCORE})
{BETTER_PROMPT_ROLE}

Why does the better framing lead to higher scores on this specific task?
Respond with one hint in less than 20 words. Focus on what the better framing emphasizes.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_ROLE_ONLY = """You are an expert in prompt optimization. Synthesize hints for writing better system_behavior instructions.

Task: {PROBLEM_DESCRIPTION}

Best-performing system_behavior instructions found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Study these top system_behavior instructions: what framing, cognitive strategy, or focus angle do the higher-scoring ones use that lower-scoring ones lack?

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Write one updated actionable hint (less than 40 words) about what framing in system_behavior scores highest on this specific task.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_ROLE_ONLY = """You are an expert in prompt optimization. Design a better system_behavior instruction.

Task: {PROBLEM_DESCRIPTION}
Task instruction (frozen): {FROZEN_PROMPT_TEXT}

[Worse system_behavior]
{WORSE_PROMPT_ROLE}
[Better system_behavior]
{BETTER_PROMPT_ROLE}
[Reflection]
{SHORT_TERM_REFLECTION}

Write a new system_behavior that takes the best aspect of both.
Rules:
- 1-2 sentences, 8-25 words total.
- You CAN start with "You are X" if you follow it with a concrete behavior.
  Pure labels without behavior (e.g., "You are an expert.") score poorly.
- Do NOT use domain-specific terms from the examples below — they are from unrelated tasks:
  "Identify the key entity being described before forming a response."
  "Weigh the broader context before committing to a specific category."
  "You are a precise reader. Prioritize the most prominent feature over peripheral details."
Output JSON: {{"system_behavior": "<new system behavior>"}}
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_ROLE_ONLY = """You are an expert in prompt optimization. Generate a new system_behavior instruction.

Task: {PROBLEM_DESCRIPTION}
Task instruction (frozen): {FROZEN_PROMPT_TEXT}

[Accumulated insight on what framing works for this task]
{LONG_TERM_REFLECTION}

[Current best system_behavior] (score: {ELITIST_SCORE})
{ELITIST_PROMPT_ROLE}

[Examples where the current system_behavior most often fails]
Each example shows the input, the wrong answer the model gave, and the correct answer.
{BAD_EXAMPLES}

Analyze these failure cases before writing:
- What type of inputs or edge cases does the current system_behavior fail to handle?
- Is there a pattern in the errors (e.g., the model misjudges a specific input type)?
- What cognitive strategy or focus angle could help the model handle these cases correctly?

Write a new system_behavior that directly addresses the identified failure pattern.
Rules:
- 1-2 sentences, 8-25 words total.
- You CAN start with "You are X" if you follow it with a concrete behavior or focus angle.
  Pure persona labels alone (e.g., "You are an expert.") are not useful.
- Short roles generalize better — avoid multi-clause chains.
- Do NOT use domain-specific terms from the examples below — they are from unrelated tasks:
  "Identify the key entity being described before forming a response."
  "Weigh the broader context before committing to a specific category."
  "You are a precise reader. Prioritize the most prominent feature over peripheral details."
Output JSON: {{"system_behavior": "<new system behavior>"}}
Output JSON data only.
"""


REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_CONSTRAINTS_ONLY = """Create diverse output_constraints variants for a fixed prompt configuration.

Task: {PROBLEM_DESCRIPTION}
Task Description (frozen): {PROMPT}
System Behavior (frozen): {ROLE}

Current output_constraints to vary from:
{CONSTRAINTS}

Create {NUM_PROMPTS} output_constraints variants with maximum diversity.
output_constraints must only contain OUTPUT FORMAT rules: what the response must or must not contain, length limits, and structural requirements.
Do NOT include in output_constraints:
- Reasoning instructions ("think step by step", "analyze X before deciding")
- Classification strategies ("use X as the default", "prefer Y when ambiguous") — those belong in system_behavior.
- Instructions that repeat what the task_description or system_behavior already says.
Do NOT repeat instructions already present in the task_description or system_behavior.
Examples of good output_constraints (output format rules only):
- "Return only the final answer. Do not include any explanation or preamble."
- "Write no more than one sentence. Use plain language."
- "Output only the requested value with no surrounding text."

Output JSON:
{{
   "prompts": [
       {{"output_constraints": "variant 1"}},
       {{"output_constraints": "variant 2"}},
       ...
       {{"output_constraints": "variant {NUM_PROMPTS}"}}
   ]
}}
Output JSON data only.
"""

REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY = """You are an expert in prompt optimization. Your task is to give hints for designing better output constraints.

Task: {PROBLEM_DESCRIPTION}
Task Description (fixed): {FROZEN_PROMPT_TEXT}
System Behavior (fixed): {FROZEN_PROMPT_ROLE}

Two output_constraints configurations were tested. The second performs better.
[Worse output_constraints] (score: {WORSE_SCORE})
{WORSE_PROMPT_CONSTRAINTS}
[Better output_constraints] (score: {BETTER_SCORE})
{BETTER_PROMPT_CONSTRAINTS}

Analyze WHY the better output FORMAT rule leads to higher scores on this task.
Consider: does stricter output brevity help? Does removing explanation noise improve parsing? Does a cleaner response structure match the evaluation metric better?
Note: output_constraints should cover FORMAT only (length, structure, what to include/exclude in the response). Do NOT suggest classification strategies or reasoning instructions — those belong in system_behavior.
Respond with one concise hint (less than 30 words).
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY = """You are an expert in prompt optimization. Your task is to give hints for designing better output constraints.

Task: {PROBLEM_DESCRIPTION}
Task Description (fixed): {FROZEN_PROMPT_TEXT}
System Behavior (fixed): {FROZEN_PROMPT_ROLE}

Best-performing output_constraints found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Study these top constraints: what format rules (brevity, structure, what to exclude) do the higher-scoring ones enforce that lower-scoring ones do not?

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Write one updated actionable hint (less than 50 words) summarizing what output constraint patterns score highest on this specific task.
Bracket the final hint with <hint> </hint>.
"""

REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_CONSTRAINTS_ONLY = """You are an expert in prompt optimization. Your task is to design better output constraints.

Task: {PROBLEM_DESCRIPTION}
Task Description (frozen): {FROZEN_PROMPT_TEXT}
System Behavior (frozen): {FROZEN_PROMPT_ROLE}

[Worse output_constraints] (score: {WORSE_SCORE})
{WORSE_PROMPT_CONSTRAINTS}
[Better output_constraints] (score: {BETTER_SCORE})
{BETTER_PROMPT_CONSTRAINTS}
[Reflection]
{SHORT_TERM_REFLECTION}

Write new output_constraints that combine the strongest aspects of both, targeting a score above {BETTER_SCORE}.
Rules:
- Must only cover OUTPUT FORMAT: what the response must/must not contain, length, structure.
- Must NOT contain reasoning instructions, chain-of-thought requirements, or classification strategies (which label to choose when uncertain) — those belong in system_behavior.
- Must NOT repeat instructions already in task_description or system_behavior.
- Keep it concise (1-2 sentences).
Output JSON: {{"output_constraints": "<new constraints>"}}
Output JSON data only.
"""

REFLECTIVEPROMPT_MUTATION_TEMPLATE_CONSTRAINTS_ONLY = """You are an expert in prompt optimization. Your task is to design new output constraints.

Task: {PROBLEM_DESCRIPTION}
Task Description (frozen): {FROZEN_PROMPT_TEXT}
System Behavior (frozen): {FROZEN_PROMPT_ROLE}

[Accumulated insight on what constraint patterns work best]
{LONG_TERM_REFLECTION}

[Current best output_constraints] (score: {ELITIST_SCORE})
{ELITIST_PROMPT_CONSTRAINTS}

[Examples where the current constraints most often fail]
Each example shows the input, the wrong answer the model gave, and the correct answer.
{BAD_EXAMPLES}

Analyze these failure cases before writing:
- Does the model output contain extra text, explanation, or preamble that hurts evaluation?
- Is the response format wrong (e.g., wrong delimiter, extra whitespace, wrong casing)?
- Would stricter length or structure rules prevent these specific errors?

Generate output_constraints that directly address the identified format failures and target a score above {ELITIST_SCORE}.
Rules:
- Must only cover OUTPUT FORMAT: what the response must/must not contain, length, structure.
- Must NOT contain reasoning instructions, chain-of-thought requirements, or classification strategies — those belong in system_behavior.
- Must NOT repeat instructions already in task_description or system_behavior.
- Try varying: length limits ("only the digit", "one sentence max"), format rules ("no preamble", "no explanation"), structural requirements.
- Keep it concise (1-2 sentences).
Output JSON: {{"output_constraints": "<new constraints>"}}
Output JSON data only.
"""



DEDUP_ROLE_TEMPLATE = """You are a prompt engineer reviewing a multi-field prompt for redundancy.

The task_description field is fixed:
TASK_DESCRIPTION: {TASK}

Review this system_behavior:
SYSTEM_BEHAVIOR: {ROLE}

Remove from SYSTEM_BEHAVIOR any content already covered by TASK_DESCRIPTION.
Specifically remove:
- Any restatement of what the task is — already in task_description
- Any label or value definitions already listed in task_description
- Any output format rules (e.g. "return only X") already in task_description
Keep only what is UNIQUE to system_behavior: cognitive strategy, reasoning approach, focus angle, default decisions when uncertain, persona framing.
If nothing unique remains, return an empty string.

Example of what to remove (translation task, for illustration only):
  task_description: "Translate the text from English to French."
  system_behavior: "You are a translator. Translate English text to French. Preserve tone."
  → Remove "Translate English text to French" (already in task). Keep "Preserve tone."

Return JSON only: {{"system_behavior": "<cleaned content or empty string>"}}
Output JSON data only."""

DEDUP_CONSTRAINTS_TEMPLATE = """You are a prompt engineer reviewing a multi-field prompt for redundancy.

The task_description and system_behavior fields are fixed:
TASK_DESCRIPTION: {TASK}
SYSTEM_BEHAVIOR: {ROLE}

Review these output_constraints:
OUTPUT_CONSTRAINTS: {CONSTRAINTS}

Remove from OUTPUT_CONSTRAINTS any content already covered by TASK_DESCRIPTION or SYSTEM_BEHAVIOR.
Specifically remove:
- Any label or value definitions already in task_description
- Any output format rules already specified in task_description
- Any reasoning instructions or decision strategies already in system_behavior
Keep only UNIQUE format rules: response length limits, structural requirements, style restrictions not already stated elsewhere.
If nothing unique remains, return an empty string.

Example of what to remove (legal task, for illustration only):
  task_description: "Identify the main obligation. Return a single sentence."
  output_constraints: "Return a single sentence stating the main obligation. Be concise."
  → Remove "Return a single sentence" (already in task). Keep "Be concise" only if it adds something new.

Return JSON only: {{"output_constraints": "<cleaned content or empty string>"}}
Output JSON data only."""
