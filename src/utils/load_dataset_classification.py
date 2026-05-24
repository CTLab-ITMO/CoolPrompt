"""Loaders for classic text-classification benchmarks.

SST-2 (sentiment), AG News (topic), TREC (question type),
SUBJ (subjectivity). All return input_data + a human-readable
label string in target.
"""

import pandas as pd
from datasets import load_dataset

SST2_LABELS = {0: "negative", 1: "positive"}
AGNEWS_LABELS = {
    0: "world", 1: "sports", 2: "business", 3: "sci/tech",
}
TREC_LABELS = {
    0: "abbreviation", 1: "entity", 2: "description",
    3: "human", 4: "location", 5: "number",
}
SUBJ_LABELS = {0: "objective", 1: "subjective"}


def _frame(texts, labels, mapping):
    return pd.DataFrame(
        {
            "input_data": [str(t) for t in texts],
            "target": [mapping[int(lbl)] for lbl in labels],
        }
    )


def load_sst2(split: str = "validation") -> pd.DataFrame:
    ds = load_dataset("stanfordnlp/sst2", split=split)
    return _frame(ds["sentence"], ds["label"], SST2_LABELS)


def load_agnews(split: str = "test") -> pd.DataFrame:
    ds = load_dataset("fancyzhx/ag_news", split=split)
    return _frame(ds["text"], ds["label"], AGNEWS_LABELS)


def load_trec(split: str = "test") -> pd.DataFrame:
    ds = load_dataset("OxAISH-AL-LLM/trec6", split=split)
    return _frame(ds["text"], ds["label"], TREC_LABELS)


def load_subj(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("SetFit/subj", split=split)
    return _frame(ds["text"], ds["label"], SUBJ_LABELS)
