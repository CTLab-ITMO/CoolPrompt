"""Prompt templates for SAPO v2."""

SEGMENTATION_PROMPT = (
    "You are an expert in prompt decomposition. Split the prompt into JSON fields: "
    "role, context, tasks, output_format. Use empty string for missing fields.\n\n"
    "Prompt:\n\"\"\"\n{prompt}\n\"\"\""
)

WEAKNESS_ANALYSIS_PROMPT = (
    "You are an expert prompt optimizer. Analyze strengths and weaknesses of the prompt by segment.\n"
    "Prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
    "Top good examples:\n{best_examples}\n\n"
    "Top bad examples:\n{worst_examples}\n\n"
    "Similar historical failures:\n{retrieved_failures}\n\n"
    "Return JSON with: weak_segments, strong_segments, recommendations."
)

FULL_REWRITE_PROMPT = (
    "You are optimizing a prompt for summarization quality, format compliance, and efficiency.\n"
    "Current prompt:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
    "Current segments:\n"
    "role={role}\ncontext={context}\ntasks={tasks}\noutput_format={output_format}\n\n"
    "Selected edit arm: {selected_arm}\n"
    "Segment confidence: {segment_confidence}\n"
    "Weak segments: {weak_segments}\n"
    "Strong segments: {strong_segments}\n"
    "Recommendations:\n{recommendations_text}\n\n"
    "Anti-pattern memory:\n{failure_memory}\n\n"
    "Generate ONE improved standalone prompt string. Keep strong segments stable. "
    "Avoid anti-patterns and generic meta text."
)

PATCH_SEGMENT_PROMPT = (
    "You are editing one segment of a prompt.\n"
    "Segment to patch: {selected_arm}\n"
    "Current segment value:\n{segment_value}\n\n"
    "Recommendation:\n{recommendation}\n\n"
    "Confidence for this segment: {segment_confidence}\n"
    "Output only the patched segment text, no extra commentary."
)

CANDIDATE_LIST_PROMPT = (
    "You are an expert prompt optimizer.\n"
    "Current prompt:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
    "Weak segments: {weak_segments}\n"
    "Strong segments: {strong_segments}\n"
    "Recommendations:\n{recommendations_text}\n\n"
    "Generate {n_candidates} diverse improved prompts. "
    "Return JSON with key 'prompts' containing a list of strings."
)
