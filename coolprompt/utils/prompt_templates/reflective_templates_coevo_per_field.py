from coolprompt.utils.prompt_templates.reflective_templates_coevo_enhanced import (
    PARAPHRASING_TEMPLATE_COEVO_ENH as PARAPHRASING_TEMPLATE_COEVO_PF,
    PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_ENH as PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_PF,
)

SHORT_TERM_REFLECTION_TEMPLATE_COEVO_PF = """You are an expert in prompt optimization. Compare two three-field configurations and identify what makes the better one score higher.

Task: {PROBLEM_DESCRIPTION}

[Worse configuration] (score: {WORSE_SCORE})
task_description:   {WORSE_PROMPT_TEXT}
system_behavior:    {WORSE_PROMPT_ROLE}
output_constraints: {WORSE_PROMPT_CONSTRAINTS}

[Better configuration] (score: {BETTER_SCORE})
task_description:   {BETTER_PROMPT_TEXT}
system_behavior:    {BETTER_PROMPT_ROLE}
output_constraints: {BETTER_PROMPT_CONSTRAINTS}

Analyze what makes the better configuration score higher. Then write three separate actionable hints, one per field (under 20 words each).
Wrap each hint in its own tags:
- task_description hint: wrap with <hint_task> </hint_task>
- system_behavior hint: wrap with <hint_role> </hint_role>
- output_constraints hint: wrap with <hint_constraints> </hint_constraints>
"""

LONG_TERM_REFLECTION_TEMPLATE_COEVO_PF = """You are an expert in prompt optimization. Synthesize patterns from the best-performing configurations found so far.

Task: {PROBLEM_DESCRIPTION}

Best configurations found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Prior accumulated per-field insights:
task_description: {PRIOR_TASK_HINT}
system_behavior:  {PRIOR_ROLE_HINT}
output_constraints: {PRIOR_CONSTRAINTS_HINT}

New per-field observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Study the top configurations above and update each per-field insight. Write one updated actionable hint per field (under 30 words each) covering the strongest pattern.
Wrap each hint in its own tags:
- task_description hint: wrap with <hint_task> </hint_task>
- system_behavior hint: wrap with <hint_role> </hint_role>
- output_constraints hint: wrap with <hint_constraints> </hint_constraints>
"""

CROSSOVER_TEMPLATE_COEVO_PF = """You are an expert in prompt optimization. Design an improved three-field prompt configuration.

Task: {PROBLEM_DESCRIPTION}

[Worse configuration] (score: {WORSE_SCORE})
task_description:   {WORSE_PROMPT_TEXT}
system_behavior:    {WORSE_PROMPT_ROLE}
output_constraints: {WORSE_PROMPT_CONSTRAINTS}

[Better configuration] (score: {BETTER_SCORE})
task_description:   {BETTER_PROMPT_TEXT}
system_behavior:    {BETTER_PROMPT_ROLE}
output_constraints: {BETTER_PROMPT_CONSTRAINTS}

[Per-field insights from comparing these configurations]
task_description:   {TASK_HINT}
system_behavior:    {ROLE_HINT}
output_constraints: {CONSTRAINTS_HINT}

Combine the strongest element from each configuration guided by the field-specific insights above.
You may take any field unchanged from either configuration, or write a new version guided by its insight.
Goal: score above {BETTER_SCORE}.

Field rules (strictly enforced):
- "task_description": WHAT to do and what output format is expected. 1–2 sentences. No reasoning instructions.
- "system_behavior": HOW to approach the task — reasoning strategy, what to prioritize, specific checks, or default decisions when input is ambiguous. Can start "You are [brief role]" ONLY if immediately followed by a concrete behavioral instruction. 1–2 sentences, 8–25 words. Must NOT repeat task_description content.
- "output_constraints": OUTPUT FORMAT rules only — length limits, structure, what to include or exclude in the response. Must NOT include reasoning instructions, decision strategies, or content already stated in the other two fields. 1–2 short rules.

Output JSON only:
{{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}}
"""

MUTATION_TEMPLATE_COEVO_PF = """You are an expert in prompt optimization. Generate a targeted mutation of the current best configuration.

Task: {PROBLEM_DESCRIPTION}

[Accumulated per-field insights on what works for this task]
task_description:   {TASK_HINT}
system_behavior:    {ROLE_HINT}
output_constraints: {CONSTRAINTS_HINT}

[Current best configuration] (score: {ELITIST_SCORE})
task_description:   {ELITIST_PROMPT_TEXT}
system_behavior:    {ELITIST_PROMPT_ROLE}
output_constraints: {ELITIST_PROMPT_CONSTRAINTS}

[Cases where the current configuration most often fails]
Each line shows: input | wrong output the model gave | correct answer.
{BAD_EXAMPLES}

Before writing, diagnose which field is responsible for these failures:
- task_description: does the instruction fail to convey the right output scope, format, or distinction between cases?
- system_behavior: does the reasoning strategy fail to handle the specific input patterns shown above, or is it biased toward certain classes?
- output_constraints: does the model produce extra text, wrong structure, or wrong format that hurts scoring?

Mutate the field(s) most responsible for the failures, guided by the per-field insights above.
Goal: score above {ELITIST_SCORE}.

Field rules (strictly enforced):
- "task_description": WHAT to do and what output format is expected. 1–2 sentences.
- "system_behavior": HOW to approach the task — reasoning strategy, what to check, default decisions for ambiguous cases. Can start "You are [brief role]" ONLY if immediately followed by a concrete behavioral instruction. 1–2 sentences, 8–25 words. Must NOT repeat task_description content.
- "output_constraints": OUTPUT FORMAT rules only — length, structure, what to include or exclude in the response. Must NOT include reasoning instructions, decision strategies, or content already stated in other fields. 1–2 short rules.

Output JSON only:
{{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}}
"""
