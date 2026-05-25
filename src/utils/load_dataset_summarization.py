"""Loader for XSum abstractive summarization."""

import pandas as pd
from datasets import load_dataset


def load_xsum(split: str = "test") -> pd.DataFrame:
    """Load XSum into input_data (document) / target (summary).

    XSum summaries are single-sentence abstractive summaries of
    BBC articles.
    """
    ds = load_dataset("EdinburghNLP/xsum", split=split)
    return pd.DataFrame(
        {
            "input_data": [str(x) for x in ds["document"]],
            "target": [str(x) for x in ds["summary"]],
        }
    )
