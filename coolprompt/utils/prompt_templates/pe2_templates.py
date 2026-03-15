PE2_REASONING_TEMPLATE = """You are a prompt engineer analyzing why a prompt fails on specific examples.

Current prompt:
{prompt}

Full template used:
{full_template}

Below are {batch_size} failure examples where the prompt produced incorrect outputs:

{examples}

For each example above, analyze:
1. What the correct output should be and what the model actually produced.
2. Why the current prompt may have led to the incorrect output.
3. Whether the prompt is misleading, ambiguous, or missing key instructions.

Provide your detailed reasoning."""


PE2_REFINEMENT_TEMPLATE = """You are a prompt engineer improving a prompt based on failure analysis.

Here is the analysis of failures with the current prompt:

{reasoning}

Current prompt:
{prompt}

Based on the analysis above, propose an improved version of the prompt that addresses the identified issues. The new prompt should be at most {max_tokens} tokens long.

Output the improved prompt between <prompt> and </prompt> tags.

<prompt>your improved prompt here</prompt>"""
