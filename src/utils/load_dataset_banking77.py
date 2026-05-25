"""Loader for the Banking77 intent classification dataset.

Banking77 (mteb/banking77) contains 3 076 customer-service
queries labelled with one of 77 banking intents.  The target
column uses the verbatim ``label_text`` strings from the
dataset (snake_case, e.g. ``activate_my_card``).
"""

import pandas as pd
from datasets import load_dataset


def banking77_labels() -> list:
    """Return the sorted list of 77 unique label strings.

    Computed lazily from the test split to avoid a network
    call at import time.  The result is cached on the module
    after the first call.
    """
    if not hasattr(banking77_labels, "_cache"):
        ds = load_dataset("mteb/banking77", split="test")
        banking77_labels._cache = sorted(
            set(ds["label_text"])
        )
    return banking77_labels._cache


def load_banking77(split: str = "test") -> pd.DataFrame:
    """Load Banking77 and return input_data / target columns.

    Args:
        split: Dataset split (default ``"test"``).

    Returns:
        DataFrame with columns ``input_data`` (str) and
        ``target`` (str, verbatim ``label_text`` value).
    """
    ds = load_dataset("mteb/banking77", split=split)
    return pd.DataFrame(
        {
            "input_data": [str(t) for t in ds["text"]],
            "target": [
                str(lbl) for lbl in ds["label_text"]
            ],
        }
    )
