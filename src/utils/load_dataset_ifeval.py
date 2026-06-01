"""Loader for the IFEval benchmark (google/IFEval).

Filters to prompts whose instruction types are all supported by
coolprompt's verifiable checkers, and encodes each prompt's
constraint spec into `target` as JSON for IFEvalMetric.
"""

import json

import pandas as pd
from datasets import load_dataset

from coolprompt.evaluator.ifeval_checkers import (
    SUPPORTED_INSTRUCTIONS,
)


def load_ifeval(max_rows: int | None = None) -> pd.DataFrame:
    """Load IFEval prompts checkable by coolprompt.

    Args:
        max_rows: Optional cap on number of rows returned.

    Returns:
        DataFrame with `input_data` (the prompt) and `target`
        (JSON: instruction_id_list + kwargs).
    """
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for ex in ds:
        ids = ex["instruction_id_list"]
        if not ids or not all(
            i in SUPPORTED_INSTRUCTIONS for i in ids
        ):
            continue
        kwargs = [
            {k: v for k, v in d.items() if v is not None}
            for d in ex["kwargs"]
        ]
        rows.append(
            {
                "input_data": str(ex["prompt"]),
                "target": json.dumps(
                    {
                        "instruction_id_list": ids,
                        "kwargs": kwargs,
                    }
                ),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=["input_data", "target"])
