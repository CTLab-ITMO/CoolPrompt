"""Benchmark configs for the new thesis datasets.

Groups: ifeval, math (gsm8k+svamp), classification
(sst2/agnews/trec/subj), russian (rucola).
"""

from src.utils.load_dataset_ifeval import load_ifeval
from src.utils.load_dataset_math import load_gsm8k, load_svamp
from src.utils.load_dataset_classification import (
    load_sst2,
    load_agnews,
    load_trec,
    load_subj,
)
from src.utils.load_dataset_russian import load_rucola

_NUMERIC_PROMPT = (
    "Solve the problem. Put only the final numeric answer "
    "inside <ans></ans> tags."
)


def _label_prompt(labels: str) -> str:
    return (
        "Classify the input. Respond with exactly one of "
        f"[{labels}] inside <ans></ans> tags."
    )


# key -> config (loader-based, like sgr_advantage_config)
BENCHMARKS = {
    "ifeval": {
        "start_prompt": (
            "Follow all instructions in the request exactly."
        ),
        "task": "generation",
        "metric": "ifeval",
        "problem_description": (
            "instruction following with verifiable constraints"
        ),
        "loader": lambda: load_ifeval(),
    },
    "gsm8k": {
        "start_prompt": _NUMERIC_PROMPT,
        "task": "generation",
        "metric": "em",
        "problem_description": "grade-school math word problems",
        "loader": lambda: load_gsm8k(),
    },
    "svamp": {
        "start_prompt": _NUMERIC_PROMPT,
        "task": "generation",
        "metric": "em",
        "problem_description": "arithmetic word problems",
        "loader": lambda: load_svamp(),
    },
    "sst2": {
        "start_prompt": _label_prompt("positive, negative"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "movie review sentiment",
        "loader": lambda: load_sst2(),
    },
    "agnews": {
        "start_prompt": _label_prompt(
            "world, sports, business, sci/tech"
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "news topic classification",
        "loader": lambda: load_agnews(),
    },
    "trec": {
        "start_prompt": _label_prompt(
            "abbreviation, entity, description, human, "
            "location, number"
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "question type classification",
        "loader": lambda: load_trec(),
    },
    "subj": {
        "start_prompt": _label_prompt("objective, subjective"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": "subjectivity classification",
        "loader": lambda: load_subj(),
    },
    "rucola": {
        "start_prompt": _label_prompt("acceptable, unacceptable"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "Russian linguistic acceptability"
        ),
        "loader": lambda: load_rucola(),
    },
}
