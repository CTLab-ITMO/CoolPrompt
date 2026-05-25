"""Benchmark configs for the new thesis datasets.

Groups: ifeval, math (gsm8k+svamp), classification
(sst2/agnews/trec/subj), russian (rucola), banking77,
anli, bbh subset.
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
from src.utils.load_dataset_banking77 import (
    load_banking77,
    banking77_labels,
)
from src.utils.load_dataset_anli import load_anli
from src.utils.load_dataset_pe2_paper import load_pe2_csv
from src.utils.load_dataset_summarization import load_xsum

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
    "xsum": {
        "start_prompt": (
            "Summarize the following text in a single concise "
            "sentence. Put only the summary inside <ans></ans> "
            "tags."
        ),
        "task": "generation",
        "metric": "rouge",
        "problem_description": (
            "abstractive single-sentence news summarization"
        ),
        "loader": lambda: load_xsum(),
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
    "banking77": {
        "start_prompt": _label_prompt(
            ", ".join(banking77_labels())
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "banking customer-intent classification (77 classes)"
        ),
        "loader": lambda: load_banking77(),
    },
    "anli": {
        "start_prompt": _label_prompt(
            "entailment, neutral, contradiction"
        ),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "adversarial natural language inference"
        ),
        "loader": lambda: load_anli(),
    },
    "bbh_boolean_expressions": {
        # targets: ['False', 'True']
        "start_prompt": _label_prompt("False, True"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: evaluate boolean expressions"
        ),
        "loader": lambda s="boolean_expressions": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
    "bbh_causal_judgement": {
        # targets: ['No', 'Yes']
        "start_prompt": _label_prompt("No, Yes"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: causal judgement reasoning"
        ),
        "loader": lambda s="causal_judgement": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
    "bbh_navigate": {
        # targets: ['No', 'Yes']
        "start_prompt": _label_prompt("No, Yes"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: navigation direction following"
        ),
        "loader": lambda s="navigate": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
    "bbh_web_of_lies": {
        # targets: ['No', 'Yes']
        "start_prompt": _label_prompt("No, Yes"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: web of lies logical deduction"
        ),
        "loader": lambda s="web_of_lies": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
    "bbh_formal_fallacies": {
        # targets: ['invalid', 'valid']
        "start_prompt": _label_prompt("invalid, valid"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: formal fallacies identification"
        ),
        "loader": lambda s="formal_fallacies": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
    "bbh_sports_understanding": {
        # targets: ['no', 'yes']
        "start_prompt": _label_prompt("no, yes"),
        "task": "classification",
        "metric": "accuracy",
        "problem_description": (
            "BBH: sports understanding plausibility"
        ),
        "loader": lambda s="sports_understanding": (
            load_pe2_csv("bbh", s, split="test")
        ),
    },
}
