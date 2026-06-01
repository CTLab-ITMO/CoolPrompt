"""Loader for RuCoLA (Russian linguistic acceptability).

Note: RussianNLP/rucola uses a legacy datasets loading
script that is no longer supported by datasets>=2.x.
We download the CSV directly via huggingface_hub and
parse it with pandas.  The validation split is
``data/dev.csv``; the test split has hidden labels
(label == -1) so the default remains ``validation``.

Field mapping (from rucola.py):
  CSV column ``acceptable`` -> ``label``
  ClassLabel names: ["unacceptable", "acceptable"]
  i.e. 0 = unacceptable, 1 = acceptable
"""

import pandas as pd
from huggingface_hub import hf_hub_download

RUCOLA_LABELS = {0: "unacceptable", 1: "acceptable"}

_SPLIT_FILES = {
    "train": "data/in_domain_train.csv",
    "validation": "data/dev.csv",
    "test": "data/test.csv",
}


def load_rucola(split: str = "validation") -> pd.DataFrame:
    """Load RuCoLA into input_data/target (label string).

    Args:
        split: One of ``"train"``, ``"validation"``,
            ``"test"``.  The test split has hidden labels
            and cannot be used for evaluation.

    Returns:
        DataFrame with columns ``input_data`` and
        ``target``.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(
            f"split must be one of {list(_SPLIT_FILES)},"
            f" got {split!r}"
        )
    if split == "test":
        raise ValueError(
            "The RuCoLA test split has hidden labels"
            " (label=-1).  Use 'train' or 'validation'."
        )
    csv_path = hf_hub_download(
        repo_id="RussianNLP/rucola",
        filename=_SPLIT_FILES[split],
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "input_data": [
                str(s) for s in df["sentence"]
            ],
            "target": [
                RUCOLA_LABELS[int(lbl)]
                for lbl in df["acceptable"]
            ],
        }
    )
