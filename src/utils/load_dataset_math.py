"""Loaders for math word-problem benchmarks (GSM8K, SVAMP)."""

import re

import pandas as pd
from datasets import load_dataset


def _final_number(text: str) -> str:
    """Extract the final numeric answer from a GSM8K answer."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else text.strip()


def load_gsm8k(split: str = "test") -> pd.DataFrame:
    """Load GSM8K into input_data/target (numeric target)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return pd.DataFrame(
        {
            "input_data": [str(q) for q in ds["question"]],
            "target": [_final_number(a) for a in ds["answer"]],
        }
    )


def load_svamp(split: str = "test") -> pd.DataFrame:
    """Load SVAMP into input_data/target (numeric target)."""
    ds = load_dataset("ChilleD/SVAMP", split=split)
    questions = [
        f"{b} {q}".strip()
        for b, q in zip(ds["Body"], ds["Question"])
    ]
    answers = [
        str(a).rstrip("0").rstrip(".") if "." in str(a) else str(a)
        for a in ds["Answer"]
    ]
    return pd.DataFrame(
        {"input_data": questions, "target": answers}
    )
