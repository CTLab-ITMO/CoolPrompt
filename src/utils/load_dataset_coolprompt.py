from datasets import load_dataset
import pandas as pd

squad_v2 = load_dataset("rajpurkar/squad_v2")
gsm8k = load_dataset("openai/gsm8k", "main")
common_gen = load_dataset("allenai/common_gen")
tweeteval = load_dataset("cardiffnlp/tweet_eval", "emotion")
xsum = load_dataset("yairfeldman/xsum")

tweeteval_emotions = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}


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


def tweeteval_preproc(sample, size: int = None):
    data = pd.DataFrame(sample)

    data["input_data"] = data["text"]
    data["target"] = data["label"].apply(lambda x: tweeteval_emotions[x])

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
                return squad_v2_preproc(squad_v2, size)
            case "gsm8k":
                return gsm8k_preproc(gsm8k, size)
            case "common_gen":
                return common_gen_preproc(common_gen, size)
            case "tweeteval":
                return tweeteval_preproc(tweeteval, size)
            case "xsum":
                return xsum_preproc(xsum, size)

    data = get_data()
    return list(data["input_data"]), list(data["target"])
