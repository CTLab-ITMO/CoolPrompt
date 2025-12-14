import os
import random
import sys
from typing import Any
from pathlib import Path
import json

import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import pandas as pd
from sklearn.model_selection import train_test_split

project_path = str(Path(__file__).resolve().parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
from config_dict import config_dict
from src.utils.load_dataset_coolprompt import tweeteval_emotions
from coolprompt.assistant import PromptTuner
from llm import OpenAITracker

model_tracker = OpenAITracker()


def create_chat_model(**kwargs):
    model = ChatOpenAI(**kwargs)
    return model_tracker.wrap_model(model)


# llm = DefaultLLM.init(vllm_engine_config={"gpu_memory_utilization": 0.95})
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=10
# )
model = "gpt-3.5-turbo"
llm = create_chat_model(
    model=model,
    temperature=0.7,
    max_tokens=4000,
    # rate_limiter=rate_limiter,
    api_key="",
    #base_url="https://openrouter.ai/api/v1",
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
            df.sample(per_class, random_state=seed)
            for _, df in data.groupby("target")
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
        data_sample = sample(preproc_data, sample_size=None)
        dataset, target = list(data_sample["input_data"]), list(
            data_sample["target"]
        )

        try:
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
                # sample_answers=True,
                # validation_size=0.5,
                evaluate=True,
                feedback=False,
            )

            result[task] = {
                "metric": {
                    "name": cfg["metric"],
                    "start_score": pt.init_metric,
                    "final_metric": pt.final_metric,
                },
                "prompt": final_prompt,
                # "samples": pt.answer_samples,
                # "cost": llm.model_stats,
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
    test("./logs/hype_exps/exp_10_hype_gsm8k.json")


if __name__ == "__main__":
    main()
