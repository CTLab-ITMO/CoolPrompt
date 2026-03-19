from typing import Optional, Tuple, List
from datasets import load_dataset as load_dataset_hf
import pandas as pd

tweeteval_emotions = {
    0: 'anger',
    1: 'joy',
    2: 'optimism',
    3: 'sadness'
}


def code_to_text_preproc(sample, size: int = None):
    """Preprocessing of CodeToText dataset"""
    data = pd.DataFrame(sample)

    def replace_docstring_text_with_empty(code: str, docstring: str) -> str:
        return code.replace(docstring, "")

    data["input_data"] = data.apply(
        lambda r: replace_docstring_text_with_empty(r["code"], r["docstring"]),
        axis=1
    )

    data['target'] = data['docstring']

    if size:
        data = data.head(size)

    return data


def concode_preproc(sample, size: int = None):
    """Preprocessing of CONCODE dataset"""
    data = pd.DataFrame(sample)

    data['input_data'] = data['nl']
    data['target'] = data['code']

    if size:
        data = data.head(size)

    return data


def medalpaca_preproc(sample, size: int = None):
    """Preprocessing of MediQA dataset"""
    data = pd.DataFrame(sample)

    data['input_data'] = data["instruction"] + "\n" + data['input']
    data['target'] = data['output']

    if size:
        data = data.head(size)

    return data


def tweeteval_preproc(sample, size: int = None):
    """Preprocessing of TweetEval (emotions) dataset"""
    data = pd.DataFrame(sample)

    data['input_data'] = data['text']
    data['target'] = data['label'].apply(
        lambda x: tweeteval_emotions[x]
    )

    if size:
        data = data.head(size)

    return data


def squad_v2_preproc(sample, size: int = None):
    """Preprocessing of SQUAD v2 dataset"""
    data = pd.DataFrame(sample)

    data["input_data"] = data["context"] + " " + data["question"]
    data["target"] = data["answers"].apply(
        lambda x: x["text"][0] if x["text"] else None
    )

    data = data.dropna()

    if size:
        data = data.head(size)

    return data


def gsm8k_preproc(sample, size: int = None):
    """Preprocessing of GSM8k dataset"""
    sample = sample['train']
    data = pd.DataFrame(sample)

    data["input_data"] = data["question"]
    data["target"] = data["answer"].apply(lambda x: x.split("####")[1].strip())

    if size:
        data = data.head(size)

    return data


def common_gen_preproc(sample, size: int = None):
    """Preprocessing of CommonGen dataset"""
    data = pd.DataFrame(sample)

    data["input_data"] = data["concepts"].apply(lambda x: str(x))

    if size:
        data = data.head(size)

    return data


def ag_news_preproc(sample, size: int = None):
    """Preprocessing of AgNews dataset"""
    data = pd.DataFrame(sample)

    data = data.rename(columns={"text": "input_data", "label": "target"})
    if size:
        data = data.head(size)

    return data


def xsum_preproc(sample, size: int = None):
    """Preprocessing of XSUM dataset"""
    data = pd.DataFrame(sample)

    data = data.rename(columns={"document": "input_data", "summary": "target"})
    if size:
        data = data.head(size)

    return data


def load_dataset(
    name: str,
    split: str,
    subset: Optional[str] = None,
    size: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """Loading preprocessed dataset

    Args:
        name (str): Name of the dataset. Supported datasets:
            ["squad_v2", "gsm8k", "common_gen", "xsum",
                "tweeteval", "mediqa", "code_to_text", "concode"]
        split (str): Split of the dataset (train or test)
        subset (Optional[str]):
            Which subset from HuggingFace to download
                (if it is needed to specify)
            Defaults to None
        size (Optional[int]): Specified size of the dataset.
            Defaults to None (full dataset size).

    Returns:
        Tuple[List[str], List[str]]: loaded dataset and targets
    """
    match name:
        case "squad_v2":
            data = load_dataset_hf("rajpurkar/squad_v2")
            match split:
                case "train": data = data[split]
                case "test": data = data['validation']
            data = squad_v2_preproc(data, size)
        case "gsm8k":
            data = load_dataset_hf("openai/gsm8k", "main")
            data = data[split]
            data = gsm8k_preproc(data, size)
        case "common_gen":
            data = load_dataset_hf("allenai/common_gen")
            data = data[split]
            data = common_gen_preproc(data, size)
        case "xsum":
            data = load_dataset_hf("yairfeldman/xsum")
            data = data[split]
            data = xsum_preproc(data, size)
        case "tweeteval":
            data = load_dataset_hf("cardiffnlp/tweet_eval", 'emotion')
            data = data[split]
            data = tweeteval_preproc(data, size)
        case "mediqa":
            data = load_dataset_hf("medalpaca/medical_meadow_mediqa")
            data = data['train']
            match split:
                case "train": data = data[:-660]
                case "test": data = data[-660:]
            data = medalpaca_preproc(data, size)
        case "code_to_text":
            data = load_dataset_hf("google/code_x_glue_ct_code_to_text", subset)
            data = data[split]
            data = code_to_text_preproc(data, size)
        case "concode":
            data = load_dataset_hf('AhmedSSoliman/CodeXGLUE-CONCODE')
            data = data[split]
            data = concode_preproc(data, size)

    return list(data["input_data"]), list(data["target"])
