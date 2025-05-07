import src.data as data
import transformers
from my_solution.stableprompt import StablePromptAgent
from pathlib import Path

model_name = "t-bank-ai/T-lite-instruct-0.1"
qconf = transformers.BitsAndBytesConfig(load_in_8bit=True)
classification_evaluator = data.TextClassificationEvaluator()
generation_evaluator = data.GenerationEvaluator()

classification_datasets = [
    data.MNLIDataset,
    data.MRDataset,
    data.QNLIDataset,
    data.SST2Dataset,
    data.TrecDataset,
    data.YahooDataset,
    data.MedQADataset,
    data.OpenbookQADataset,
]

generation_datasets = [
    data.GSM8KDataset,
    data.MathDataset,
    data.SamsumDataset,
]


def make_classification_agent(log_file, dataset):
    sp_agent = StablePromptAgent(
        task="classification",
        log_file=log_file,
        target_model_name=model_name,
        agent_model_name=model_name,
        quantization_config=qconf,
        dataset=dataset,
        evaluator=classification_evaluator,
        metric="f1",
    )
    return sp_agent


def make_generation_agent(log_file, dataset):
    sp_agent = StablePromptAgent(
        task="generation",
        log_file=log_file,
        target_model_name=model_name,
        agent_model_name=model_name,
        quantization_config=qconf,
        dataset=dataset,
        evaluator=generation_evaluator,
        metric="f1",
    )
    return sp_agent


def full_testing():
    path = Path("logs/")
    path.mkdir(parents=True, exist_ok=True)

    for dataset in classification_datasets:
        log_file_name = dataset.__class__.__name__ + "_log.txt"
        result_file_name = dataset.__class__.__name__ + "_result.txt"
        agent = make_classification_agent(path / log_file_name, dataset)
        agent.train()
        agent.test(path / result_file_name)
    for dataset in generation_datasets:
        log_file_name = dataset.__class__.__name__ + "_log.txt"
        result_file_name = dataset.__class__.__name__ + "_result.txt"
        agent = make_generation_agent(path / log_file_name, dataset)
        agent.train()
        agent.test(path / result_file_name)


def test_on_one_dataset():
    path = Path("logs/")
    path.mkdir(parents=True, exist_ok=True)

    dataset = data.SST2Dataset
    log_file_name = dataset.__class__.__name__ + "_log.txt"
    result_file_name = dataset.__class__.__name__ + "_result.txt"
    agent = make_classification_agent(path / log_file_name, dataset)
    agent.train()
    agent.test(path / result_file_name)


if __name__ == "__main__":
    test_on_one_dataset()
