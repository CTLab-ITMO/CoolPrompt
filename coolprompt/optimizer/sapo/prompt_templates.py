"""Meta-prompts used by the segment-based SAPO optimizer."""

SEGMENTATION_TEMPLATE = """You are an expert prompt engineer.
Decompose the prompt below into four segments. Return JSON only with string
fields: role, context, tasks, output_format. Use an empty string when a segment
is absent. Do not invent text that is not present in the prompt.

Prompt:
\"\"\"
{prompt}
\"\"\"
"""

WEAKNESS_ANALYSIS_TEMPLATE = """You are an expert prompt engineer.
Analyze the prompt using the strongest and weakest examples below. Return JSON
only with these fields:
- weak_segments: a list containing only role, context, tasks, output_format
- strong_segments: a list containing only role, context, tasks, output_format
- recommendations: an object mapping weak segment names to concise actions

Prompt:
\"\"\"
{prompt}
\"\"\"

Current segments:
Role: {role}
Context: {context}
Tasks: {tasks}
Output format: {output_format}

Best examples:
{best_examples}

Worst examples:
{worst_examples}
"""

CANDIDATE_GENERATION_TEMPLATE = """You are an expert prompt engineer.
Generate {n_candidates} diverse, standalone improved versions of the current
prompt. Modify the weak segments according to the recommendations and preserve
the strong segments. Do not copy dataset examples into a prompt. Do not add an
input placeholder: CoolPrompt appends each input separately at runtime.

Return JSON only as {{"prompts": ["candidate 1", "candidate 2"]}}.

Current prompt:
\"\"\"
{current_prompt}
\"\"\"

Segments:
Role: {role}
Context: {context}
Tasks: {tasks}
Output format: {output_format}

Weak segments: {weak_segments}
Strong segments: {strong_segments}
Recommendations:
{recommendations}
"""

