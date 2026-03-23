"""Configuration for all PE2 paper benchmarks.

69 tasks across 4 categories: math (2), bbh (29),
instruction_induction (24), counterfactual-evaluation (13).
One task (arithmetic_base10) has no prompt.md — we use a
plain arithmetic fallback.
"""

from src.utils.load_dataset_pe2_paper import (
    load_pe2_csv,
    load_pe2_prompt,
)

# -----------------------------------------------------------
# Instruction Induction metric mapping.
# CoolPrompt's "em" only works for numeric targets.
# For text targets we use "bertscore" as approximation.
# Only "diff" and "sum" have numeric targets → use "em".
# -----------------------------------------------------------
# Tasks with numeric targets that work with "em".
# Everything else uses "bertscore" as text approximation.
NUMERIC_TASKS = {
    # math — all numeric
    "multiarith", "gsm8k",
    # bbh
    "multistep_arithmetic_two", "object_counting",
    # ii
    "diff", "sum",
    # cf — only pure-decimal bases work with em
    "arithmetic_base8", "arithmetic_base9",
    "arithmetic_base10",
    # base11 (digits 0-9,A) and base16 (hex) have letter
    # digits that break extract_number_from_text → bertscore
}


def _metric_for(subtask: str) -> str:
    """Return 'em' for numeric tasks, 'bertscore' otherwise."""
    return "em" if subtask in NUMERIC_TASKS else "bertscore"

# -----------------------------------------------------------
# Category subtask lists
# -----------------------------------------------------------
MATH_SUBTASKS = [
    "multiarith",
    "gsm8k",
]

BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "data_understanding_generative",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "movie_recommendation_generative",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

II_SUBTASKS = [
    "active_to_passive",
    "antonyms",
    "cause_and_effect",
    "common_concept",
    "diff",
    "first_word_letter",
    "informal_to_formal",
    "larger_animal",
    "letters_list",
    "negation",
    "num_to_verbal",
    "orthography_starts_with",
    "rhymes",
    "second_word_letter",
    "sentence_similarity",
    "sentiment",
    "singular_to_plural",
    "sum",
    "synonyms",
    "taxonomy_animal",
    "translation_en-de",
    "translation_en-es",
    "translation_en-fr",
    "word_in_context",
]

CF_SUBTASKS = [
    "arithmetic_base8",
    "arithmetic_base9",
    "arithmetic_base10",
    "arithmetic_base11",
    "arithmetic_base16",
    "chess_cf",
    "chess_original",
    "syntax_osv",
    "syntax_ovs",
    "syntax_sov",
    "syntax_svo",
    "syntax_vos",
    "syntax_vso",
]

# -----------------------------------------------------------
# Problem descriptions per category
# -----------------------------------------------------------
CATEGORY_DESCRIPTIONS = {
    "math": "math problem solving",
    "bbh": "BIG-Bench Hard reasoning",
    "instruction_induction": "instruction induction",
    "counterfactual-evaluation": "counterfactual evaluation",
}


def build_pe2_paper_config(ii_cf_split: int = 0):
    """Build the full config dict for PE2 paper benchmarks.

    Args:
        ii_cf_split: Which split (0-4) to use for
            instruction_induction and
            counterfactual-evaluation tasks.

    Returns:
        Dict mapping ``"category/subtask"`` to config
        entries compatible with the test runner.
    """
    config = {}

    # --- Math (2 tasks) ---
    for subtask in MATH_SUBTASKS:
        key = f"math/{subtask}"
        config[key] = {
            "category": "math",
            "subtask": subtask,
            "start_prompt": load_pe2_prompt(
                "math", subtask
            ),
            "task": "generation",
            "metric": _metric_for(subtask),
            "problem_description": "math problem solving",
            "loader": lambda s=subtask: load_pe2_csv(
                "math", s
            ),
        }

    # --- BBH (29 tasks) ---
    for subtask in BBH_SUBTASKS:
        key = f"bbh/{subtask}"
        desc = subtask.replace("_", " ")
        config[key] = {
            "category": "bbh",
            "subtask": subtask,
            "start_prompt": load_pe2_prompt(
                "bbh", subtask
            ),
            "task": "generation",
            "metric": _metric_for(subtask),
            "problem_description": f"BIG-Bench Hard: {desc}",
            "loader": lambda s=subtask: load_pe2_csv(
                "bbh", s
            ),
        }

    # --- Instruction Induction (24 tasks) ---
    for subtask in II_SUBTASKS:
        key = f"ii/{subtask}"
        desc = subtask.replace("_", " ").replace("-", " ")
        config[key] = {
            "category": "instruction_induction",
            "subtask": subtask,
            "start_prompt": load_pe2_prompt(
                "instruction_induction",
                subtask,
                ii_cf_split,
            ),
            "task": "generation",
            "metric": _metric_for(subtask),
            "problem_description": (
                f"instruction induction: {desc}"
            ),
            "loader": lambda s=subtask, sp=ii_cf_split: (
                load_pe2_csv(
                    "instruction_induction", s,
                    ii_cf_split=sp,
                )
            ),
        }

    # --- Counterfactual Evaluation (13 tasks) ---
    for subtask in CF_SUBTASKS:
        key = f"cf/{subtask}"
        desc = subtask.replace("_", " ")
        config[key] = {
            "category": "counterfactual-evaluation",
            "subtask": subtask,
            "start_prompt": load_pe2_prompt(
                "counterfactual-evaluation",
                subtask,
                ii_cf_split,
            ),
            "task": "generation",
            "metric": _metric_for(subtask),
            "problem_description": (
                f"counterfactual evaluation: {desc}"
            ),
            "loader": lambda s=subtask, sp=ii_cf_split: (
                load_pe2_csv(
                    "counterfactual-evaluation", s,
                    ii_cf_split=sp,
                )
            ),
        }

    return config
