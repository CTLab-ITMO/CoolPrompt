"""Configuration for SGR advantage benchmarks.

Phase 1: High-baseline PE2 paper tasks where PE2 shows
    little improvement (start_score > 0.85, delta < 0.02).
    SGR's structured refinement may help here.

Phase 2: Custom multi-constraint generation tasks designed
    to produce heterogeneous errors — exactly where SGR's
    error categorization adds value.
"""

import json
from pathlib import Path

from src.utils.load_dataset_pe2_paper import (
    load_pe2_csv,
    load_pe2_prompt,
)

# -----------------------------------------------------------
# Phase 1: High-baseline PE2 task subset
# -----------------------------------------------------------
# Tasks with start_score > 0.85 AND PE2 delta < +0.02
# from v2 benchmark results.
PHASE1_KEYS = [
    "bbh/boolean_expressions",
    "bbh/causal_judgement",
    "bbh/data_understanding_generative",
    "bbh/navigate",
    "cf/arithmetic_base9",
    "cf/arithmetic_base10",
    "cf/arithmetic_base16",
    "cf/chess_cf",
    "cf/chess_original",
    "math/multiarith",
]

# Metric mapping (same as pe2_paper_config)
NUMERIC_TASKS = {
    "multiarith", "gsm8k",
    "multistep_arithmetic_two", "object_counting",
    "diff", "sum",
    "arithmetic_base8", "arithmetic_base9",
    "arithmetic_base10",
}


def _metric_for(subtask: str) -> str:
    return "em" if subtask in NUMERIC_TASKS else "bertscore"


def _parse_key(key: str):
    """Parse 'category/subtask' into (category, subtask)."""
    category_short, subtask = key.split("/", 1)
    category_map = {
        "math": "math",
        "bbh": "bbh",
        "ii": "instruction_induction",
        "cf": "counterfactual-evaluation",
    }
    return category_map[category_short], subtask


def build_phase1_config(ii_cf_split: int = 0):
    """Build config for Phase 1 high-baseline tasks.

    Returns:
        Dict mapping task keys to config entries.
    """
    config = {}
    for key in PHASE1_KEYS:
        category, subtask = _parse_key(key)
        desc = subtask.replace("_", " ")

        loader_kwargs = {}
        if category in (
            "instruction_induction",
            "counterfactual-evaluation",
        ):
            loader_kwargs["ii_cf_split"] = ii_cf_split

        config[key] = {
            "category": category,
            "subtask": subtask,
            "start_prompt": load_pe2_prompt(
                category, subtask,
                *([ii_cf_split] if loader_kwargs else []),
            ),
            "task": "generation",
            "metric": _metric_for(subtask),
            "problem_description": (
                f"{category.replace('-', ' ')}: {desc}"
            ),
            "loader": (
                lambda s=subtask, c=category, kw=dict(
                    loader_kwargs
                ): load_pe2_csv(
                    c, s, split="dev", **kw,
                )
            ),
            "train_loader": (
                lambda s=subtask, c=category, kw=dict(
                    loader_kwargs
                ): load_pe2_csv(
                    c, s, split="train", **kw,
                )
            ),
        }
    return config


# -----------------------------------------------------------
# Phase 2: Custom multi-constraint tasks
# -----------------------------------------------------------
DATA_DIR = Path(__file__).parent / "sgr_advantage_data"


def _load_json_dataset(filename: str):
    """Load a JSON dataset file into a DataFrame."""
    import pandas as pd

    path = DATA_DIR / filename
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# Custom LLM-as-judge templates for each task.
# Each checks multiple constraints simultaneously.
# Template vars: {metric_ceil}, {request}, {response}

SUMMARIZATION_TEMPLATE = (
    "You are evaluating a constrained summarization.\n\n"
    "Source text:\n{request}\n\n"
    "Summary produced:\n{response}\n\n"
    "Rate the summary from 1 to {metric_ceil} based on "
    "ALL of these constraints:\n"
    "1. Length: exactly 2-3 sentences\n"
    "2. Tone: formal/academic (no slang, contractions, "
    "or casual language)\n"
    "3. Key fact: includes the most important factual "
    "claim from the source\n\n"
    "Scoring guide:\n"
    "- 1-3: Fails 2+ constraints\n"
    "- 4-6: Meets 1 constraint well, fails others\n"
    "- 7-8: Meets 2 constraints well\n"
    "- 9-10: Meets all 3 constraints\n\n"
    "Output ONLY a single integer score:"
)

EXTRACTION_TEMPLATE = (
    "You are evaluating structured data extraction.\n\n"
    "Source text:\n{request}\n\n"
    "Extraction produced:\n{response}\n\n"
    "Rate from 1 to {metric_ceil} based on ALL constraints:\n"
    "1. Format: output uses exactly "
    "'Name: X, Location: Y, Date: YYYY-MM-DD'\n"
    "2. Accuracy: extracted values match the source text\n"
    "3. Completeness: all entities present in the source "
    "are extracted\n\n"
    "Scoring guide:\n"
    "- 1-3: Wrong format AND missing/wrong entities\n"
    "- 4-6: Correct format but inaccurate/incomplete, "
    "or accurate but wrong format\n"
    "- 7-8: Correct format with minor extraction errors\n"
    "- 9-10: Perfect format and fully accurate extraction\n\n"
    "Output ONLY a single integer score:"
)

CREATIVE_TEMPLATE = (
    "You are evaluating constrained creative writing.\n\n"
    "Topic prompt:\n{request}\n\n"
    "Response produced:\n{response}\n\n"
    "Rate from 1 to {metric_ceil} based on ALL constraints:\n"
    "1. Structure: exactly 2 paragraphs\n"
    "2. Opening: first sentence is a question\n"
    "3. Closing: last sentence is a call to action\n\n"
    "Scoring guide:\n"
    "- 1-3: Fails 2+ constraints\n"
    "- 4-6: Meets 1 constraint, fails others\n"
    "- 7-8: Meets 2 constraints\n"
    "- 9-10: Meets all 3 constraints\n\n"
    "Output ONLY a single integer score:"
)

INSTRUCTION_TEMPLATE = (
    "You are evaluating multi-step instruction following.\n\n"
    "Question:\n{request}\n\n"
    "Response produced:\n{response}\n\n"
    "Rate from 1 to {metric_ceil} based on ALL constraints:\n"
    "1. Structure: starts with a direct answer\n"
    "2. Reasons: lists exactly 3 supporting reasons as "
    "numbered bullets\n"
    "3. Conclusion: ends with a single-sentence conclusion\n\n"
    "Scoring guide:\n"
    "- 1-3: Missing 2+ structural elements\n"
    "- 4-6: Has answer but wrong bullet count or "
    "no conclusion\n"
    "- 7-8: All elements present with minor issues\n"
    "- 9-10: Perfect structure and content quality\n\n"
    "Output ONLY a single integer score:"
)

CLASSIFICATION_EXPLAIN_TEMPLATE = (
    "You are evaluating format-sensitive classification "
    "with explanation.\n\n"
    "Review:\n{request}\n\n"
    "Response produced:\n{response}\n\n"
    "Rate from 1 to {metric_ceil} based on ALL constraints:\n"
    "1. Format: output is exactly "
    "'Label: X\\nReason: Y'\n"
    "2. Label accuracy: sentiment label (Positive/Negative/"
    "Neutral) is correct\n"
    "3. Explanation: reason is exactly one sentence and "
    "genuinely explains the label\n\n"
    "Scoring guide:\n"
    "- 1-3: Wrong format AND wrong label\n"
    "- 4-6: Correct format but wrong label, or correct "
    "label but wrong format\n"
    "- 7-8: Correct format and label, reason has minor "
    "issues\n"
    "- 9-10: Perfect format, correct label, clear "
    "explanation\n\n"
    "Output ONLY a single integer score:"
)


# Phase 2 task definitions
PHASE2_TASKS = {
    "custom/constrained_summarization": {
        "start_prompt": (
            "Summarize the following text."
        ),
        "task": "generation",
        "metric": "llm_as_judge",
        "problem_description": (
            "constrained summarization: 2-3 sentences, "
            "formal tone, include key fact"
        ),
        "llm_as_judge_criteria": "constraint_compliance",
        "llm_as_judge_custom_templates": {
            "constraint_compliance": SUMMARIZATION_TEMPLATE,
        },
        "llm_as_judge_metric_ceil": 10,
        "data_file": "constrained_summarization.json",
    },
    "custom/structured_extraction": {
        "start_prompt": (
            "Extract entities from the following text."
        ),
        "task": "generation",
        "metric": "llm_as_judge",
        "problem_description": (
            "structured data extraction: "
            "Name/Location/Date format"
        ),
        "llm_as_judge_criteria": "constraint_compliance",
        "llm_as_judge_custom_templates": {
            "constraint_compliance": EXTRACTION_TEMPLATE,
        },
        "llm_as_judge_metric_ceil": 10,
        "data_file": "structured_extraction.json",
    },
    "custom/constrained_creative": {
        "start_prompt": (
            "Write a short response about the "
            "following topic."
        ),
        "task": "generation",
        "metric": "llm_as_judge",
        "problem_description": (
            "constrained creative writing: 2 paragraphs, "
            "starts with question, ends with call to action"
        ),
        "llm_as_judge_criteria": "constraint_compliance",
        "llm_as_judge_custom_templates": {
            "constraint_compliance": CREATIVE_TEMPLATE,
        },
        "llm_as_judge_metric_ceil": 10,
        "data_file": "constrained_creative.json",
    },
    "custom/multi_step_instruction": {
        "start_prompt": (
            "Answer the following question."
        ),
        "task": "generation",
        "metric": "llm_as_judge",
        "problem_description": (
            "multi-step instruction following: answer + "
            "3 numbered reasons + conclusion"
        ),
        "llm_as_judge_criteria": "constraint_compliance",
        "llm_as_judge_custom_templates": {
            "constraint_compliance": INSTRUCTION_TEMPLATE,
        },
        "llm_as_judge_metric_ceil": 10,
        "data_file": "multi_step_instruction.json",
    },
    "custom/classification_explanation": {
        "start_prompt": (
            "Classify the sentiment of the following "
            "review."
        ),
        "task": "generation",
        "metric": "llm_as_judge",
        "problem_description": (
            "format-sensitive classification with "
            "explanation: Label + Reason format"
        ),
        "llm_as_judge_criteria": "constraint_compliance",
        "llm_as_judge_custom_templates": {
            "constraint_compliance": (
                CLASSIFICATION_EXPLAIN_TEMPLATE
            ),
        },
        "llm_as_judge_metric_ceil": 10,
        "data_file": "classification_explanation.json",
    },
}


def build_phase2_config():
    """Build config for Phase 2 custom tasks.

    Returns:
        Dict mapping task keys to config entries.
    """
    config = {}
    for key, cfg in PHASE2_TASKS.items():
        data_file = cfg["data_file"]
        config[key] = {
            **{k: v for k, v in cfg.items()
               if k != "data_file"},
            "loader": lambda f=data_file: (
                _load_json_dataset(f)
            ),
            "train_loader": None,
        }
    return config
