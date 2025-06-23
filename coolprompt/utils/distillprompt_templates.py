"""
Prompt templates for the DistillPrompt optimization process.

This module stores all the f-string templates used by the PromptTransformer
to generate, aggregate, compress, and distill prompts.
"""

import textwrap

AGGREGATION_PROMPT = textwrap.dedent("""
    Below are several prompts intended for the same task:

    {formatted_prompts}

    Your task is to generate one clear and concise prompt that captures
    the general idea, overall objective, and key instructions conveyed
    by all of the above prompts. Focus on the shared purpose and main
    concepts without including specific examples or extraneous details.

    Return only the new prompt, and enclose it with <START> and <END> tags.
""")

COMPRESSION_PROMPT = textwrap.dedent("""
    I want to compress the following zero-shot classifier prompt
    into a shorter prompt of 2–3 concise sentences that capture
    its main objective and key ideas from any examples.

    Current prompt: {candidate_prompt}

    Steps:
    1. Identify the main task or objective.
    2. Extract the most important ideas illustrated by the examples.
    3. Combine these insights into a brief, coherent prompt.

    Return only the new prompt, and enclose it with <START> and <END> tags.
""")

DISTILLATION_PROMPT = textwrap.dedent("""
    You are an expert prompt engineer.

    Current instruction prompt: {candidate_prompt}

    Training examples:
    {sample_string}

    Task:
    Analyze the current prompt and training examples to understand
    common strengths and weaknesses. Learn the general insights
    and patterns without copying any example text. Rewrite the
    instruction prompt to improve clarity and effectiveness while
    maintaining the original intent. Do not include any extraneous
    explanation or details beyond the revised prompt.

    Return only the new prompt, and enclose it with <START> and <END> tags.
""")

GENERATION_PROMPT = textwrap.dedent("""
    You are an expert in prompt analysis with exceptional comprehension skills.

    Below is my current instruction prompt:
    {candidate_prompt}

    On the train dataset, this prompt scored {train_score:.3f}
    (with 1.0 being the maximum).

    Please analyze the prompt's weaknesses and generate an improved
    version that refines its clarity, focus, and instructional quality.
    Do not assume any data labels—focus solely on the quality of the prompt.

    Return only the improved prompt, and enclose it with <START> and <END> tags.
    Improved prompt:
""")

REWRITER_PROMPT = (
    "Generate a variation of the following prompt while keeping the semantic meaning.\n\n"
    "Input: {candidate_prompt}\n\n"
    "Output:"
)
