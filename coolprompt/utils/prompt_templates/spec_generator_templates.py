SPEC_FROM_PROMPT_TEMPLATE = """\
You are an expert NLP task analyst.
Your job is NOT to answer the task prompt.
Your job is to analyze it and produce a structured task specification
that will be used to generate synthetic training/evaluation examples.

<task_prompt>
{prompt}
</task_prompt>

Produce a detailed specification with exactly these fields:

- domain: the subject-matter area of the task
- task_type: one of: classification | generation | summarization | QA | translation | extraction | evaluation | other
- task_summary: one sentence describing exactly what the model must do
- io_format:
    - input_description: format and content of the input
    - output_description: format and content of the expected output
    - input_constraints: list of input formatting rules such as length, casing, punctuation, language, or structure
    - output_constraints: list of output formatting rules such as label-only output, JSON shape, length, casing, or no extra text
- key_skills: 4-8 atomic capabilities required to solve the task
- constraints: 3-6 hard rules that every valid answer must follow
- typical_errors: 3-6 common mistakes a language model may make on this task
- corner_cases: 4-8 tricky but realistic patterns, each formatted as:
    - name: short identifier
    - description: what makes this case hard
    - example_hint: a concrete hint at what such an input looks like
- language: primary language of the task; default to English if unclear
- label_set: exhaustive list of valid labels for classification tasks; null for all other task types
- additional_notes: practical notes useful for a synthetic data generator

The JSON MUST have this exact top-level structure:
{{
  "domain": "string",
  "task_type": "generation",
  "task_summary": "string",
  "io_format": {{
    "input_description": "string",
    "output_description": "string",
    "input_constraints": ["string"],
    "output_constraints": ["string"]
  }},
  "key_skills": ["string"],
  "constraints": ["string"],
  "typical_errors": ["string"],
  "corner_cases": [
    {{
      "name": "string",
      "description": "string",
      "example_hint": "string"
    }}
  ],
  "language": "English",
  "label_set": null,
  "additional_notes": null,
  "matched_dataset": null
}}

Important:
- Return the COMPLETE TaskSpecification object.
- Do NOT return only input_description and output_description.
- Do NOT put input_description or output_description at the top level.
- input_description and output_description MUST be inside io_format.
- Use null for label_set unless task_type is classification.
- Use null for matched_dataset if no known dataset is detected.

Be concrete and specific to THIS task prompt.
Do not give generic NLP advice.
Do not attempt to solve the task itself.
Return only valid JSON matching the TaskSpecification schema.
Do not include markdown, comments, or explanations.
"""

SPEC_FROM_PROMPT_AND_EXAMPLES_TEMPLATE = """\
You are an expert NLP task analyst.
Your job is NOT to answer the task prompt.
Your job is to analyze it together with the provided examples and produce
a structured task specification that will be used to generate synthetic
training/evaluation examples.

<task_prompt>
{prompt}
</task_prompt>

<input_output_examples>
{examples}
</input_output_examples>

Produce a detailed specification with exactly these fields:

- domain: the subject-matter area of the task
- task_type: one of: classification | generation | summarisation | QA | translation | extraction | evaluation | other
- task_summary: one sentence describing exactly what the model must do
- io_format:
    - input_description: format and content of the input, inferred from examples when possible
    - output_description: format and content of the expected output, inferred from examples when possible
    - input_constraints: input formatting rules observed or implied by the examples
    - output_constraints: output formatting rules observed or implied by the examples
- key_skills: 4-8 atomic capabilities required to solve the task
- constraints: 3-6 hard rules that every valid answer must follow
- typical_errors: 3-6 common mistakes a language model may make on this task
- corner_cases: 4-8 tricky but realistic patterns, each formatted as:
    - name: short identifier
    - description: what makes this case hard
    - example_hint: a concrete hint, preferably based on patterns visible in or extrapolated from the examples
- language: primary language of the task, inferred from examples if not stated in the prompt
- label_set: exhaustive list of valid labels for classification tasks, inferred from both the prompt and examples; if examples show only a subset, mention this in additional_notes; null for all other task types
- additional_notes: practical notes useful for a synthetic data generator, including any contradictions between the prompt and examples

Ground your analysis in the examples.
Be concrete and specific to THIS task prompt.
Do not give generic NLP advice.
Do not attempt to solve the task itself.
Return only valid JSON matching the TaskSpecification schema.
Do not include markdown, comments, or explanations.
"""

SPEC_REGULAR_CLASSIFICATION_TEMPLATE = """\
You are a synthetic data generator for NLP tasks.

TASK SPECIFICATION:
  Domain          : {domain}
  Task summary    : {task_summary}
  Input format    : {input_description}
  Output format   : {output_description}
  Valid labels    : {label_set}
  Key skills      : {key_skills}
  Constraints     : {constraints}
  Language        : {language}
  Notes           : {additional_notes}

Generate exactly {num_samples} diverse input-output examples that cover
the skills [{focused_skills}] and respect the listed constraints.

Each example MUST have:
  - "input" : a realistic input sample
  - "output": the correct label (one of {label_set})

Return ONLY a JSON object: {{"examples": [{{"input": "...", "output": "..."}}]}}
"""

SPEC_CORNER_CLASSIFICATION_TEMPLATE = """\
You are a synthetic data generator specialising in hard, adversarial cases.

TASK SPECIFICATION:
  Domain          : {domain}
  Task summary    : {task_summary}
  Input format    : {input_description}
  Output format   : {output_description}
  Valid labels    : {label_set}
  Typical errors  : {typical_errors}
  Language        : {language}

TARGET CORNER-CASE PATTERN:
  Name            : {corner_name}
  Description     : {corner_description}
  Generation hint : {corner_hint}

Generate exactly {num_samples} examples that specifically exhibit the
corner-case pattern above.  Make them realistic but clearly tricky.

Each example MUST have:
  - "input" : a realistic but challenging task input
  - "output": exactly one valid label from: {label_set}

Return ONLY a JSON object: {{"examples": [{{"input": "...", "output": "..."}}]}}
"""

SPEC_REGULAR_GENERATION_TEMPLATE = """\
You are a synthetic data generator for NLP tasks.

TASK SPECIFICATION:
  Domain          : {domain}
  Task summary    : {task_summary}
  Input format    : {input_description}
  Output format   : {output_description}
  Key skills      : {key_skills}
  Constraints     : {constraints}
  Language        : {language}
  Notes           : {additional_notes}

Generate exactly {num_samples} diverse input-output examples that cover
the skills [{focused_skills}] and respect the listed constraints.

Each example MUST have:
  - "input" : a realistic task input
  - "output": the expected correct output for that input

Return ONLY a JSON object: {{"examples": [{{"input": "...", "output": "..."}}]}}
"""

SPEC_CORNER_GENERATION_TEMPLATE = """\
You are a synthetic data generator specialising in hard, adversarial cases.

TASK SPECIFICATION:
  Domain          : {domain}
  Task summary    : {task_summary}
  Input format    : {input_description}
  Output format   : {output_description}
  Typical errors  : {typical_errors}
  Constraints     : {constraints}
  Language        : {language}

TARGET CORNER-CASE PATTERN:
  Name            : {corner_name}
  Description     : {corner_description}
  Generation hint : {corner_hint}

Generate exactly {num_samples} examples that specifically exhibit the
corner-case pattern above.  Inputs must be challenging; outputs must be
correct despite the difficulty.

Each example MUST have:
  - "input" : a realistic but challenging task input
  - "output": the expected correct output for that input

Return ONLY a JSON object: {{"examples": [{{"input": "...", "output": "..."}}]}}
"""
