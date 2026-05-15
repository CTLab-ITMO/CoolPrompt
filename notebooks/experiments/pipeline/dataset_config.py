import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_dir, "../../../")))
sys.path.append(os.path.abspath(os.path.join(_dir, "../../../src")))

from datasets import load_dataset
from coolprompt.utils.enums import Task
from utils.load_dataset_coolprompt import (
    squad_v2,
    squad_v2_preproc,
    gsm8k,
    gsm8k_preproc,
    common_gen,
    common_gen_preproc,
    xsum,
    xsum_preproc,
)

_MEDIQA_OPT_OFFSET = 130

DATASETS_CONFIG = {
    "tweet_eval": {
        "path": "cardiffnlp/tweet_eval",
        "task": Task.CLASSIFICATION,
        "metric": "f1",
        "input_field": "text",
        "target_field": "label",
        "subset": "sentiment",
        "initial_task_description": "Classify the sentiment of the text. Return only the number: 0 for negative, 1 for neutral, or 2 for positive.",
        "initial_system_behavior": "Consider the overall tone of the text. When signals are mixed or ambiguous, prefer neutral (1) — reserve positive (2) and negative (0) for clearly expressed emotions.",
        "initial_output_constraints": "Return only the number (0, 1, or 2). Do not include explanations or any other text.",
        "description": "Classifying the sentiment of social media posts (tweets) as positive, negative, or neutral.",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "task": Task.GENERATION,
        "metric": "em",
        "input_field": "question",
        "target_field": "answer",
        "subset": "main",
        "initial_task_description": "Solve the math problem.",
        "initial_system_behavior": "Show your reasoning step by step.",
        "initial_output_constraints": "State the final answer as a single number on the last line. Verify each arithmetic step before moving to the next.",
        "description": "Solving grade school math word problems involving multi-step reasoning.",
    },
    "squad_v2": {
        "path": "rajpurkar/squad_v2",
        "task": Task.GENERATION,
        "metric": "bertscore",
        "input_field": "question",
        "target_field": "answers",
        "initial_task_description": "Answer the question based on the context.",
        "initial_system_behavior": "Answer based only on the provided text.",
        "initial_output_constraints": "If the context does not contain enough information to answer, respond with 'I cannot determine this from the given context.' Otherwise, give the shortest direct answer.",
        "description": "Answering questions based on a provided text passage (context).",
    },
    "common_gen": {
        "path": "allenai/common_gen",
        "task": Task.GENERATION,
        "metric": "bertscore",
        "input_field": "concepts",
        "target_field": "target",
        "initial_task_description": "Write a fluent sentence that uses all the given words.",
        "initial_system_behavior": "Write naturally and coherently.",
        "initial_output_constraints": "Use every provided word in the sentence.",
        "description": "Generating a coherent sentence that includes all words from a given list of concepts.",
    },
    "xsum": {
        "path": "yairfeldman/xsum",
        "task": Task.GENERATION,
        "metric": "bertscore",
        "input_field": "document",
        "target_field": "summary",
        "initial_task_description": "Summarize the article in one sentence.",
        "initial_system_behavior": "Focus on the main point.",
        "initial_output_constraints": "Output exactly one sentence. Keep it under 25 words and use neutral, factual language.",
        "description": "Creating a concise one-sentence summary of a news article.",
    },
    "mediqa": {
        "path": "medalpaca/medical_meadow_mediqa",
        "task": Task.GENERATION,
        "metric": "bertscore",
        "input_field": "question",
        "target_field": "answer",
        "initial_task_description": "Answer the medical question.",
        "initial_system_behavior": "Be accurate and thorough.",
        "initial_output_constraints": "Answer in 1-3 sentences using plain clinical language. Do not include citations or reference numbers.",
        "description": "Medical question answering: provide accurate, thorough, evidence-based answers.",
    },
}


def load_train_data(dataset_name, config, num_samples=200):
    print(f"loading: {dataset_name}")

    if dataset_name == "squad_v2":
        data = squad_v2_preproc(squad_v2["train"], size=num_samples)
        return list(data["input_data"]), list(data["target"])
    if dataset_name == "gsm8k":
        data = gsm8k_preproc(gsm8k["train"], size=num_samples)
        return list(data["input_data"]), list(data["target"])
    if dataset_name == "common_gen":
        data = common_gen_preproc(common_gen["train"], size=num_samples)
        return list(data["input_data"]), list(data["target"])
    if dataset_name == "xsum":
        data = xsum_preproc(xsum["train"], size=num_samples)
        return list(data["input_data"]), list(data["target"])

    subset = config.get("subset")
    if dataset_name == "mediqa":
        dataset = (
            load_dataset(config["path"], subset, split="train")
            if subset
            else load_dataset(config["path"], split="train")
        )
        inputs, targets = [], []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inp = sample.get(
                "instruction", sample.get("input", sample.get("question", ""))
            )
            tgt = sample.get("output", sample.get("answer", ""))
            if inp and tgt:
                inputs.append(inp)
                targets.append(tgt)
        return inputs, targets

    dataset = (
        load_dataset(config["path"], subset, split="train")
        if subset
        else load_dataset(config["path"], split="train")
    )
    inputs, targets = [], []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        inp = str(sample.get(config["input_field"], ""))
        tgt = str(sample.get(config["target_field"], ""))
        if inp and tgt:
            inputs.append(inp)
            targets.append(tgt)
    return inputs, targets


def load_eval_data(dataset_name, config, num_samples, seed, full_test=False):
    if full_test:
        num_samples = None
        seed = None
    print(
        f"eval: {dataset_name} ({'full' if full_test else f'max {num_samples}'}, seed={seed})"
    )

    if dataset_name == "squad_v2":
        data = squad_v2_preproc(
            squad_v2["validation"], size=num_samples, seed=seed
        )
        inputs, targets = list(data["input_data"]), list(data["target"])
    elif dataset_name == "gsm8k":
        data = gsm8k_preproc(gsm8k["test"], size=num_samples, seed=seed)
        inputs, targets = list(data["input_data"]), list(data["target"])
    elif dataset_name == "common_gen":
        data = common_gen_preproc(
            common_gen["validation"], size=num_samples, seed=seed
        )
        inputs, targets = list(data["input_data"]), list(data["target"])
    elif dataset_name == "xsum":
        data = xsum_preproc(xsum["test"], size=num_samples, seed=seed)
        inputs, targets = list(data["input_data"]), list(data["target"])
    elif dataset_name == "mediqa":
        subset = config.get("subset")
        dataset = (
            load_dataset(config["path"], subset)
            if subset
            else load_dataset(config["path"])
        )
        split_name = (
            "test"
            if "test" in dataset
            else ("validation" if "validation" in dataset else "train")
        )
        print(f"  mediqa split: {split_name}")
        ds_split = dataset[split_name]
        if split_name == "train":
            offset = _MEDIQA_OPT_OFFSET if full_test else 100
            ds_split = ds_split.select(range(offset, len(ds_split)))
        if seed is not None:
            ds_split = ds_split.shuffle(seed=seed)
        limit = len(ds_split) if num_samples is None else num_samples
        inputs, targets = [], []
        for i in range(len(ds_split)):
            if len(inputs) >= limit:
                break
            sample = ds_split[i]
            inp = sample.get(
                "instruction", sample.get("input", sample.get("question", ""))
            )
            tgt = sample.get("output", sample.get("answer", ""))
            if inp and tgt:
                inputs.append(inp)
                targets.append(tgt)
    elif dataset_name == "tweet_eval":
        dataset = load_dataset(config["path"], config["subset"], split="test")
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        limit = len(dataset) if num_samples is None else num_samples
        inputs, targets = [], []
        for i in range(len(dataset)):
            if len(inputs) >= limit:
                break
            sample = dataset[i]
            inp = str(sample.get(config["input_field"], ""))
            tgt = str(sample.get(config["target_field"], ""))
            if inp and tgt:
                inputs.append(inp)
                targets.append(tgt)
    else:
        subset = config.get("subset")
        dataset = (
            load_dataset(config["path"], subset, split="train")
            if subset
            else load_dataset(config["path"], split="train")
        )
        offset = _MEDIQA_OPT_OFFSET if full_test else 100
        dataset = dataset.select(range(offset, len(dataset)))
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        inputs, targets = [], []
        for sample in dataset:
            inp = str(sample.get(config["input_field"], ""))
            tgt = str(sample.get(config["target_field"], ""))
            if inp and tgt:
                inputs.append(inp)
                targets.append(tgt)

    print(f"loaded {len(inputs)} samples")
    return inputs, targets
