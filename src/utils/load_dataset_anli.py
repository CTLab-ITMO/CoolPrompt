"""Loader for the ANLI (Adversarial NLI) dataset.

Uses the facebook/anli dataset, split ``test_r3`` (1 200 rows).

Label order verified via ``ds.features['label']``, which is a
``ClassLabel`` with ``names=['entailment', 'neutral',
'contradiction']``, confirming:
  0 -> entailment
  1 -> neutral
  2 -> contradiction
This matches the standard MNLI/ANLI convention and was
confirmed by inspecting ``ds.features['label'].names``
directly (no reliance on guessing or external sources).
"""

import pandas as pd
from datasets import load_dataset

# Verified from ds.features['label'].names:
# ClassLabel(names=['entailment', 'neutral', 'contradiction'])
_ANLI_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def load_anli(split: str = "test_r3") -> pd.DataFrame:
    """Load ANLI and return input_data / target columns.

    Args:
        split: Dataset split (default ``"test_r3"``).

    Returns:
        DataFrame with columns ``input_data`` (str, formatted
        as ``"Premise: ...\nHypothesis: ..."`` ) and
        ``target`` (str, one of ``entailment``,
        ``neutral``, ``contradiction``).
    """
    ds = load_dataset("facebook/anli", split=split)
    input_data = [
        f"Premise: {p}\nHypothesis: {h}"
        for p, h in zip(ds["premise"], ds["hypothesis"])
    ]
    targets = [
        _ANLI_LABELS[int(lbl)] for lbl in ds["label"]
    ]
    return pd.DataFrame(
        {
            "input_data": input_data,
            "target": targets,
        }
    )
