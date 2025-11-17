import os
import random
import sys
from typing import Any
from pathlib import Path
import json

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import pandas as pd
from sklearn.model_selection import train_test_split

project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
from config_dict import config_dict
from src.utils.load_dataset_coolprompt import ag_labels
from coolprompt.assistant import PromptTuner
from coolprompt.language_model.llm import DefaultLLM, OpenAITracker

model_tracker = OpenAITracker()


def create_chat_model(**kwargs):
    model = ChatOpenAI(**kwargs)
    return model_tracker.wrap_model(model)


# llm = DefaultLLM.init(vllm_engine_config={"gpu_memory_utilization": 0.95})
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=10
# )
llm = create_chat_model(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=4000,
    openai_api_key="",
)
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
        data_sample = sample(preproc_data, sample_size=10)
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
            "samples": pt.answer_samples,
        }

    return result


def test(path: str | Path) -> None:
    with open(path, "w") as f:
        result = run_hype_dataset()
        print("Saving to", os.path.abspath(path))
        json.dump(result, f)
        print(f"Successfully wrote to {path}")


def main():
    test("./logs/open_squad_test_2.json")


if __name__ == "__main__":
    main()
