"""Loader for PE2 paper benchmark datasets.

Reads CSV data directly from the PE2 repository at
``../../PE2/data/`` relative to this file's grandparent.
"""

from pathlib import Path

import pandas as pd

PE2_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "PE2" / "data"
)


def load_pe2_csv(
    category: str,
    subtask: str,
    split: str = "test",
    ii_cf_split: int = 0,
) -> pd.DataFrame:
    """Load a PE2 paper CSV dataset.

    Args:
        category: One of "math", "bbh",
            "instruction_induction",
            "counterfactual-evaluation".
        subtask: Subtask name, e.g. "multiarith",
            "boolean_expressions", "antonyms".
        split: One of "train", "dev", "test".
        ii_cf_split: 0-4 for instruction_induction and
            counterfactual-evaluation tasks (ignored for
            math and bbh).

    Returns:
        DataFrame with columns ``input_data`` and
        ``target``.
    """
    base = PE2_DATA_DIR / category / subtask

    if category in ("instruction_induction",
                     "counterfactual-evaluation"):
        csv_path = base / str(ii_cf_split) / f"{split}.csv"
    else:
        csv_path = base / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Handle non-standard column layouts
    if subtask == "cause_and_effect":
        df = df.rename(
            columns={"cause": "input_data",
                      "effect": "target"}
        )
    elif subtask == "common_concept":
        df = df.rename(
            columns={"items": "input_data",
                      "concept": "target"}
        )
    else:
        if "input" in df.columns:
            df = df.rename(
                columns={"input": "input_data"}
            )
        if "label" in df.columns:
            df = df.rename(
                columns={"label": "target"}
            )
        if "output" in df.columns:
            df = df.rename(
                columns={"output": "target"}
            )

    df["input_data"] = df["input_data"].astype(str)
    df["target"] = df["target"].astype(str)

    return df[["input_data", "target"]]


def load_pe2_prompt(
    category: str,
    subtask: str,
    ii_cf_split: int = 0,
) -> str:
    """Load the initial prompt for a PE2 paper task.

    Args:
        category: Dataset category.
        subtask: Subtask name.
        ii_cf_split: Split index for II/CF tasks.

    Returns:
        Prompt string.
    """
    base = PE2_DATA_DIR / category / subtask

    if category in ("instruction_induction",
                     "counterfactual-evaluation"):
        prompt_path = base / str(ii_cf_split) / "prompt.md"
    else:
        prompt_path = base / "prompt.md"

    if prompt_path.exists():
        return prompt_path.read_text().strip()

    # Fallback defaults
    if category in ("math", "bbh"):
        return "Let's think step by step."
    return ""
