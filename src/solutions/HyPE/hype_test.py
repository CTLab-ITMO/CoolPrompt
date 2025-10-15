import random
import sys
from typing import Any
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
from config_dict import config_dict
from src.utils.load_dataset_coolprompt import ag_labels
from coolprompt.assistant import PromptTuner
from coolprompt.language_model.llm import DefaultLLM

llm = DefaultLLM.init()
pt = PromptTuner(llm)


def manage_ag_news(data: pd.DataFrame, max_imbalance: float = 0.6):
    if set(data["target"].unique()).issubset(set(ag_labels)):
        class_proportions = data["target"].value_counts(normalize=True)
        if class_proportions.max() > max_imbalance:
            return None
        else:
            return data


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


def run_hype_dataset() -> dict[str, Any]:
    result = {}

    for task, cfg in config_dict.items():
        data_train, data_val = (
            cfg["data"]["train"],
            cfg["data"][cfg["test_name"]],
        )
        preproc_data = cfg["preproc"](data_val)
        data_sample = sample(preproc_data, sample_size=100)
        dataset, target = list(data_sample["input_data"]), list(
            data_sample["target"]
        )

        final_prompt = pt.run(
            cfg["start_prompt"],
            cfg["task"],
            dataset,
            target,
            "hype",
            cfg["metric"],
            cfg["problem_description"],
            verbose=2,
            train_as_test=True,
            sample_answers=True,
        )

        result[task] = {
            "metric": {
                "name": cfg["metric"],
                "start_score": pt.init_metric,
                "final_metric": pt.final_metric,
            },
            "prompt": final_prompt,
            "samples": (
                pt.answer_samples if hasattr(pt, "answer_samples") else None
            ),
        }

    return result


def test(path: str | Path) -> None:
    with open(path, "w") as f:
        result = run_hype_dataset()
        json.dump(result, f)


def main():
    test("./logs/test_1.json")


if __name__ == "__main__":
    main()
