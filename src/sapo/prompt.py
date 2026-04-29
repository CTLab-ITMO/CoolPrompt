"""Prompt templates used by SAPO optimization pipelines."""

BASE_META_PROMPT = (
    "You are an expert prompt engineer. You are given a prompt that is used"
    "to instruct a language model to perform a specific task.\n"
    "Current prompt:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
    "There are a list of tuples with incorrect answers which includes "
    "(input, reference, model_answer): {bad_examples}"
    "Your task to analyze and recognize weaknesses of this prompt by incorrect answers. "
    "By your analytics please generate an improved version of the prompt."
)

CANDIDATE_GEN_TEMPLATE = (
    "You are an expert prompt engineer. Based on the following prompt:\n"
    "\"\"\"\n{base_prompt}\n\"\"\"\n"
    "Generate {n} different improved versions of this prompt. "
    "Each version should be a standalone prompt that can be used for the same task. "
    "Output the prompts as a JSON list of strings, for example: [\"version 1\", \"version 2\", ...]. "
    "Do not add any explanations, only the JSON."
)

WEAKNESS_ANALYSIS_TEMPLATE = (
    "Given the prompt:\n"
    "\"\"\"\n{prompt}\n\"\"\"\n\n"
    "It was used to generate responses for the following examples. "
    "For each example, the input query, reference answer, and model response are provided.\n"
    "Examples with the lowest BERTScore (worst 5):\n"
    "{bad_examples_str}\n\n"
    "Analyze why the prompt failed on these examples and suggest a specific recommendation "
    "for improving it. The recommendation should be concise and actionable. "
    "Output only the recommendation text."
)

IMPROVEMENT_TEMPLATE = (
    "Original prompt:\n"
    "\"\"\"\n{prompt}\n\"\"\"\n\n"
    "Improvement recommendation:\n"
    "{recommendation}\n\n"
    "Create a new, improved version of the prompt that incorporates this recommendation. "
    "Output only the new prompt text."
)

__all__ = [
    "BASE_META_PROMPT",
    "CANDIDATE_GEN_TEMPLATE",
    "WEAKNESS_ANALYSIS_TEMPLATE",
    "IMPROVEMENT_TEMPLATE",
]
