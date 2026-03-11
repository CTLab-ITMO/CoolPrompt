import os
import sys
from typing import Any
from pathlib import Path
import json

import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import pandas as pd

project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
from config_dict import config_dict
from src.utils.load_dataset_coolprompt import tweeteval_emotions
from coolprompt.assistant import PromptTuner

# llm = DefaultLLM.init(vllm_engine_config={"gpu_memory_utilization": 0.95})
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=10
)
model = "gpt-4o-mini"
llm = ChatOpenAI(
    model=model,
    temperature=0.7,
    max_completion_tokens=4000,
    max_retries=5,
    rate_limiter=rate_limiter,
    api_key="",
    extra_body={
        "allowed_providers": ["google-vertex", "azure"],
    },
    base_url="https://openrouter.ai/api/v1",
)
pt = PromptTuner(llm)


def sample(
    data: pd.DataFrame,
    sample_size: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    if sample_size is None:
        return data

    if set(data["target"].unique()).issubset(set(tweeteval_emotions)):
        min_class_size = data["target"].value_counts().min()
        per_class = min(sample_size // len(tweeteval_emotions), min_class_size)

        balanced_parts = [
            df.sample(per_class, random_state=seed) for _, df in data.groupby("target")
        ]
        return pd.concat(balanced_parts).reset_index(drop=True)
    else:
        return data.sample(sample_size, random_state=seed)


def run_hype_dataset() -> dict[str, Any]:
    result = {"model": model}

    for task, cfg in config_dict.items():
        data_train, data_val = (
            cfg["data"]["train"],
            cfg["data"][cfg["test_name"]],
        )
        preproc_data = cfg["preproc"](data_val)
        data_sample = sample(preproc_data, sample_size=10)
        dataset, target = list(data_sample["input_data"]), list(data_sample["target"])

        try:
            final_prompt = pt.run(
                cfg["start_prompt"],
                cfg["task"],
                dataset,
                target,
                "hyper",
                cfg["metric"],
                cfg["problem_description"],
                verbose=2,
                train_as_test=True,
                feedback=False,
            )

            result[task] = {
                "metric": {
                    "name": cfg["metric"],
                    "start_score": pt.init_metric,
                    "final_metric": pt.final_metric,
                },
                "prompt": final_prompt,
            }
        except Exception as e:
            print(f"!!!!EXCEPTION: {str(e)}!!!!")
            result[task] = {"exception": str(e)}

    return result


def test(path: str | Path) -> None:
    with open(path, "w") as f:
        result = run_hype_dataset()
        print("Saving to", os.path.abspath(path))
        json.dump(result, f)
        print(f"Successfully wrote to {path}")


def main():
    test("./logs/result.json")


if __name__ == "__main__":
    main()
