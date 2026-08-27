PARAPHRASING_TEMPLATE_COEVO_ENH = """Create {NUM_PROMPTS} diverse initial variants of the following three-field configuration.

Task: {PROBLEM_DESCRIPTION}

Seed configuration:
task_description:   {PROMPT}
system_behavior:    {ROLE}
output_constraints: {CONSTRAINTS}

Rules for each variant:
- Vary at least two fields meaningfully from the seed.
- "task_description": change wording, directness, or how the output format is stated — preserve the task intent.
- "system_behavior": vary the reasoning strategy, cognitive angle, or focus area. Can start "You are [role]" only if immediately followed by a concrete behavioral instruction. 8–25 words. Must NOT restate task content.
- "output_constraints": vary format rules — length limits, structure, what to include or exclude. Must NOT include reasoning instructions or decision strategies (those belong in system_behavior).
- Each variant must differ meaningfully from the others.

Output JSON only:
{{
  "prompts": [
    {{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}},
    {{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}},
    ...
  ]
}}
Output JSON data only.
"""

SHORT_TERM_REFLECTION_TEMPLATE_COEVO_ENH = """You are an expert in prompt optimization. Compare two three-field configurations and identify what makes the better one score higher.

Task: {PROBLEM_DESCRIPTION}

[Worse configuration] (score: {WORSE_SCORE})
task_description:   {WORSE_PROMPT_TEXT}
system_behavior:    {WORSE_PROMPT_ROLE}
output_constraints: {WORSE_PROMPT_CONSTRAINTS}

[Better configuration] (score: {BETTER_SCORE})
task_description:   {BETTER_PROMPT_TEXT}
system_behavior:    {BETTER_PROMPT_ROLE}
output_constraints: {BETTER_PROMPT_CONSTRAINTS}

Analyze each field separately:
- task_description: what difference in wording, directness, or format specification matters?
- system_behavior: what difference in reasoning strategy, focus area, or decision rule matters?
- output_constraints: what difference in format rule, length limit, or exclusion matters?

Then write ONE combined actionable hint (under 30 words) identifying the most impactful change.
Wrap the hint with <hint> </hint>.
"""

LONG_TERM_REFLECTION_TEMPLATE_COEVO_ENH = """You are an expert in prompt optimization. Synthesize patterns from the best-performing configurations found so far.

Task: {PROBLEM_DESCRIPTION}

Best configurations found so far (ranked by score, best first):
{TOP_PROMPTS_HISTORY}

Prior accumulated insight:
{PRIOR_LONG_TERM_REFLECTION}

New per-field observations from recent comparisons:
{NEW_SHORT_TERM_REFLECTIONS}

Study the top configurations above. Identify what distinguishes the highest-scoring ones:
- task_description: what phrasing, directness, or output-format specification appears in the highest-scoring configs?
- system_behavior: what reasoning strategy, focus angle, or decision heuristic appears in the highest-scoring configs?
- output_constraints: what format rule — strictness of brevity, structure, or exclusions — correlates with higher scores?

Write ONE updated actionable hint (under 50 words) covering the strongest pattern across all three fields.
Wrap the hint with <hint> </hint>.
"""

CROSSOVER_TEMPLATE_COEVO_ENH = """You are an expert in prompt optimization. Design an improved three-field prompt configuration.

Task: {PROBLEM_DESCRIPTION}

[Worse configuration] (score: {WORSE_SCORE})
task_description:   {WORSE_PROMPT_TEXT}
system_behavior:    {WORSE_PROMPT_ROLE}
output_constraints: {WORSE_PROMPT_CONSTRAINTS}

[Better configuration] (score: {BETTER_SCORE})
task_description:   {BETTER_PROMPT_TEXT}
system_behavior:    {BETTER_PROMPT_ROLE}
output_constraints: {BETTER_PROMPT_CONSTRAINTS}

[Key insight from comparing these configurations]
{SHORT_TERM_REFLECTION}

Combine the strongest element from each configuration. You may take any field unchanged from either configuration, or write a new version of a field guided by the insight above.
Goal: score above {BETTER_SCORE}.

Field rules (strictly enforced):
- "task_description": WHAT to do and what output format is expected. 1–2 sentences. No reasoning instructions.
- "system_behavior": HOW to approach the task — reasoning strategy, what to prioritize, specific checks, or default decisions when input is ambiguous. Can start "You are [brief role]" ONLY if immediately followed by a concrete behavioral instruction. 1–2 sentences, 8–25 words. Must NOT repeat task_description content.
- "output_constraints": OUTPUT FORMAT rules only — length limits, structure, what to include or exclude in the response. Must NOT include reasoning instructions, decision strategies, or content already stated in the other two fields. 1–2 short rules.

Output JSON only:
{{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}}
"""

MUTATION_TEMPLATE_COEVO_ENH = """You are an expert in prompt optimization. Generate a targeted mutation of the current best configuration.

Task: {PROBLEM_DESCRIPTION}

[Accumulated insight on what works for this task]
{LONG_TERM_REFLECTION}

[Current best configuration] (score: {ELITIST_SCORE})
task_description:   {ELITIST_PROMPT_TEXT}
system_behavior:    {ELITIST_PROMPT_ROLE}
output_constraints: {ELITIST_PROMPT_CONSTRAINTS}

[Cases where the current configuration most often fails]
Each line shows: input | wrong output the model gave | correct answer.
{BAD_EXAMPLES}

Before writing, diagnose which field is responsible for these failures:
- task_description issue: does the instruction fail to convey the right output scope, format, or distinction between cases?
- system_behavior issue: does the reasoning strategy fail to handle the specific input patterns shown above, or is it biased toward certain classes?
- output_constraints issue: does the model produce extra text, wrong structure, or wrong format that hurts scoring?

Mutate the field(s) most responsible for the failures. The other fields may stay the same or be improved moderately.
Goal: score above {ELITIST_SCORE}.

Field rules (strictly enforced):
- "task_description": WHAT to do and what output format is expected. 1–2 sentences.
- "system_behavior": HOW to approach the task — reasoning strategy, what to check, default decisions for ambiguous cases. Can start "You are [brief role]" ONLY if immediately followed by a concrete behavioral instruction. 1–2 sentences, 8–25 words. Must NOT repeat task_description content.
- "output_constraints": OUTPUT FORMAT rules only — length, structure, what to include or exclude in the response. Must NOT include reasoning instructions, decision strategies, or content already stated in other fields. 1–2 short rules.

Output JSON only:
{{"task_description": "...", "system_behavior": "...", "output_constraints": "..."}}
"""

PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_ENH = """Generate a concise initial three-field prompt configuration for the following task.

Task: {PROBLEM_DESCRIPTION}

Output a JSON object with exactly three fields:
- "task_description": What the model should do and how to return the answer (1–2 sentences). Be specific about the output format (e.g., a label, a number, a single sentence, the exact expected structure).
- "system_behavior": How the model should approach the task — one concrete reasoning principle or focus area (1 sentence, 8–20 words). Must be a behavioral instruction, not just a role title.
  Good: "Check for ambiguous cases before deciding." / "Focus on the key entity before forming a response."
  Bad: "You are an expert." — vague, no behavioral instruction.
- "output_constraints": Output format rules only — what to include or exclude in the response (1–2 short rules, under 15 words total).
  Must NOT include reasoning instructions or decision strategies — those belong in system_behavior.

Keep all fields brief and generic. These are starting points to be refined by the optimizer.
Output JSON only.
"""
