from datasets import load_dataset as load_dataset_hf
import pandas as pd


ag_labels = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3,
}

tweeteval_emotions = {
    0: 'anger',
    1: 'joy',
    2: 'optimism',
    3: 'sadness'
}


def medalpaca_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data['input_data'] = data["instruction"] + "\n" + data['input']
    data['target'] = data['output']

    if size:
        data = data.head(size)

    return data


def tweeteval_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data['input_data'] = data['text']
    data['target'] = data['label'].apply(
        lambda x: tweeteval_emotions[x]
    )

    if size:
        data = data.head(size)

    return data


def squad_v2_preproc(sample, size: int = None):
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
    data = pd.DataFrame(sample)

    data["input_data"] = data["question"]
    data["target"] = data["answer"].apply(lambda x: x.split("####")[1].strip())

    if size:
        data = data.head(size)

    return data


def common_gen_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data["input_data"] = data["concepts"].apply(lambda x: str(x))

    if size:
        data = data.head(size)

    return data


def ag_news_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data = data.rename(columns={"text": "input_data", "label": "target"})
    if size:
        data = data.head(size)

    return data


def xsum_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data = data.rename(columns={"document": "input_data", "summary": "target"})
    if size:
        data = data.head(size)

    return data


def load_dataset(name: str, size: int = None):
    def get_data():
        match name:
            case "squad_v2":
                squad_v2 = load_dataset_hf("rajpurkar/squad_v2", split="validation")
                return squad_v2_preproc(squad_v2, size)
            case "gsm8k":
                gsm8k = load_dataset_hf("openai/gsm8k", "main", split="test")
                return gsm8k_preproc(gsm8k, size)
            case "common_gen":
                common_gen = load_dataset_hf("allenai/common_gen", split="validation")
                return common_gen_preproc(common_gen, size)
            case "ag_news":
                ag_news = load_dataset_hf("fancyzhx/ag_news", split="test")
                return ag_news_preproc(ag_news, size)
            case "xsum":
                xsum = load_dataset_hf("yairfeldman/xsum", split="validation")
                return xsum_preproc(xsum, size)
            case "medalpaca":
                medalpaca = load_dataset_hf("medalpaca/medical_meadow_mediqa", split='train')
                return medalpaca_preproc(medalpaca, size)
            case "tweeteval":
                tweeteval = load_dataset_hf("cardiffnlp/tweet_eval", "emotion", split='train')
                return tweeteval_preproc(tweeteval, size)

    data = get_data()
    return list(data["input_data"]), list(data["target"])
