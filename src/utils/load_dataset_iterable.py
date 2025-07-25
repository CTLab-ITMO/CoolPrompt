import random
from typing import Iterable, Tuple
from datasets import load_dataset

MULTITASK_TASKS = ["bbh"]


def load_mnli(split: str, **kwargs) -> Tuple[Iterable, Iterable]:
    """Loads MNLI dataset

    Args:
        split (str): dataset split mode (train or test)
    """

    mnli = load_dataset("nyu-mll/multi_nli", "plain_text", split=split)
    dataset = [
        f"Premise: {p} Hypothesis: {h}"
        for p, h in zip(mnli["premise"], mnli["hypothesis"])
    ]
    targets = mnli["label"]
    return dataset, targets


def load_ethos(split: str, **kwargs) -> Tuple[Iterable, Iterable]:
    """Loads Ethos dataset

    Args:
        split (str): dataset split mode (train or test)
    """

    ethos = load_dataset("ethos", "binary", split=split)
    dataset = ethos["text"]
    targets = ethos["label"]
    return dataset, targets


def load_openbookqa(split: str, **kwargs) -> Tuple[Iterable, Iterable]:
    """Loads OpenbookQA dataset

    Args:
        split (str): dataset split mode (train or test)
    """

    openbookqa = load_dataset("allenai/openbookqa", split=split)
    dataset = openbookqa["question_stem"]
    targets = openbookqa["answerKey"]
    return dataset, targets


def load_bbh(split: str, subtest: str, **kwargs) -> Tuple[Iterable, Iterable]:
    """Loads BBH dataset with specified subtest

    Args:
        split (str): dataset split mode (train or test)
        subtest (str): BBH subtest name ('causal_judgement', etc.)
    Raises:
        ValueError: if subtest name is not specified.
    """

    if subtest is None:
        raise ValueError(
            "You must specify a subtest for using the BBH dataset"
        )
    bbh = load_dataset("lukaemon/bbh", subtest, split=split)
    dataset = bbh["inputs"]
    targets = bbh["targets"]
    return dataset, targets


def load_dataset_iterable(
    dataset_name: str,
    split: str = "test",
    sample_size: int = None,
    seed: int = 42,
):
    """Loads dataset with provided name and split mode. Uses random
    subset with `sample_size` instances if provided.

    Args:
        dataset_name (str): full dataset name with subtest name after '/'
            (if benchmark is multitask). Must be one of: 'mnli', 'ethos',
            'openbookqa', 'bbh/subtest' (specify subtest name).
        split (str): dataset split mode (train or test).
        sample_size (int, optional): specifies the size of the subset.
            Will use full dataset if not provided.
        seed (int): seed for generating indices for random subset.
            Defaults to 42.
    Raises:
        ValueError: if dataset name is not supported or subtest name is not
            specified for multitask dataset.
    """
    dataset_name = dataset_name.lower()
    if "/" in dataset_name:
        bench_name, subtest_name = dataset_name.split("/", 1)
    else:
        bench_name, subtest_name = dataset_name, None

    dataset2loading = {
        "mnli": load_mnli,
        "ethos": load_ethos,
        "openbookqa": load_openbookqa,
        "bbh": load_bbh,
    }

    if bench_name not in dataset2loading:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader = dataset2loading[bench_name]

    if bench_name in MULTITASK_TASKS:
        dataset, target = loader(split=split, subtest=subtest_name)
    else:
        dataset, target = loader(split=split)

    if sample_size is not None:
        total_size = len(dataset)
        n = min(sample_size, total_size)

        rng = random.Random(seed)
        indices = rng.sample(range(total_size), n)

        dataset_list, target_list = list(dataset), list(target)

        dataset = [dataset_list[i] for i in indices]
        target = [target_list[i] for i in indices]

    return dataset, target
