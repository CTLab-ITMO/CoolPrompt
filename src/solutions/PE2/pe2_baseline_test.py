import os
import random
import sys
from typing import Any
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_openai import ChatOpenAI

project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
hype_path = str(Path(__file__).resolve().parent.parent / "HyPE")
sys.path.append(hype_path)
from config_dict import config_dict
from src.utils.load_dataset_coolprompt import ag_labels
from coolprompt.assistant import PromptTuner

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("CP_OPENAI_KEY"),
    max_retries=10,
)
pt = PromptTuner(llm)

BASELINE_DATASETS = ["squad_v2", "ag_news"]


def sample(
    data: pd.DataFrame,
    sample_size: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    if sample_size is not None:
        if set(data["target"].unique()).issubset(set(ag_labels)):
            _, data_sample = train_test_split(
                data,
                train_size=sample_size,
                stratify=data["target"],
                random_state=seed,
            )
        else:
            rng = random.Random(seed)
            total_size = len(data)
            n = min(sample_size, total_size)
            indices = rng.sample(range(total_size), n)
            data_sample = data.iloc[indices]
        return data_sample
    return data


def run_pe2_baseline() -> dict[str, Any]:
    result = {}

    for task, cfg in config_dict.items():
        if task not in BASELINE_DATASETS:
            continue

        test_split = cfg.get("test_name", "validation")
        preproc_data = cfg["preproc"](cfg["data"][test_split])
        data_sample = sample(preproc_data, sample_size=50)
        dataset, target = list(data_sample["input_data"]), list(
            data_sample["target"]
        )

        final_prompt = pt.run(
            cfg["start_prompt"],
            cfg["task"],
            dataset,
            target,
            "pe2",
            cfg["metric"],
            cfg["problem_description"],
            verbose=2,
            train_as_test=True,
            train_steps=2,
        )

        result[task] = {
            "metric": {
                "name": cfg["metric"],
                "start_score": pt.init_metric,
                "final_metric": pt.final_metric,
            },
            "prompt": final_prompt,
        }

    return result


def main():
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/pe2_baseline_test_1.json", "w") as f:
        result = run_pe2_baseline()
        json.dump(result, f)


if __name__ == "__main__":
    main()
