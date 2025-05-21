import requests
from vllm import SamplingParams


HEADERS = {"Content-Type": "application/json"}


class Infer:
    """Inference helper class for vllm server"""

    def __init__(self, model_name,
                 server_url="http://localhost:8000/v1/completions",
                 model_generate_args={}):
        self.model_name = model_name
        self.server_url = server_url
        self.model_generate_args = model_generate_args

    def __call__(self, prompt, label_id=None):
        """Label is needed to ensure label <-> prompt match"""
        result = vllm_infer(
                prompt,
                self.model_name,
                server_url=self.server_url,
                **self.model_generate_args
        )
        if len(result) == 1:
            result = result[0]
        return result, label_id

class LLMWrapper:
    """Inference helper class for vllm.LLM class"""

    def __init__(self, model,
                 model_generate_args={}):
        self.model = model
        self.model_generate_args = model_generate_args

    def __call__(self, prompt, **model_gen_args) -> list[str] | str:
        
        model_gen_args = model_gen_args or {}
        model_gen_args = self.model_generate_args | model_gen_args
        
        sampling_params = SamplingParams(**model_gen_args)
        
        answers = self.model.generate(
            prompts=prompt, sampling_params=sampling_params, use_tqdm=False
        ) # list[RequestOutput]

        answers = answers[0] # single prompt scenario

        texts = [output.text for output in answers.outputs]

        if len(texts) == 1: # n = 1 scenario
            return texts[0]
        
        return texts


def vllm_infer(
    prompt,
    model_name,
    stop_token_ids,
    server_url="http://localhost:8000/v1/completions",
    temperature=0.0,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0,
    timeout=100,
):
    """Про параметры читать тут: https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams"""
    with requests.Session() as session:
        payload = {
            "prompt": prompt,
            "model": model_name,
            "temperature": temperature,
            "n": n,
            "top_p": top_p,
            "stop": stop,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        response = session.post(
            server_url, json=payload, headers=HEADERS, timeout=timeout
        )
        completions = response.json().get("choices", [])
        return [completion["text"] for completion in completions]


from src.data.base.datasets.multi_task_dataset import BaseMultiTaskDataset
from src.data.base.datasets.classification_dataset import BaseClassificationDataset
from src.data.base.datasets.dataset import BaseDataset


from src.data.classification import MNLIDataset, MRDataset, QNLIDataset, SST2Dataset, TrecDataset, YahooDataset
from src.data.generation import GSM8KDataset, MathDataset, SamsumDataset
from src.data.multi_task import BBHDataset, NaturalInstructionsDataset
from src.data.qa import MedQADataset, OpenbookQADataset



TASK_TO_DS = {
    "sst-2": SST2Dataset,
    
    "natural_instructions/task021": NaturalInstructionsDataset,
    "natural_instructions/task050": NaturalInstructionsDataset,
    "natural_instructions/task069": NaturalInstructionsDataset,
    
    "math": MathDataset,
    "gsm8k": GSM8KDataset,

    "yahoo": YahooDataset,
    "trec": TrecDataset,
    "mr": MRDataset,
    
    "openbookqa": OpenbookQADataset,
    "samsum": SamsumDataset,
    
    "qnli": QNLIDataset,
    "mnli": MNLIDataset,
    "medqa": MedQADataset,
    
        
    "bbh/boolean_expressions" : BBHDataset,
    "bbh/hyperbaton" : BBHDataset,
    "bbh/temporal_sequences" : BBHDataset,
    "bbh/object_counting" : BBHDataset,
    "bbh/disambiguation_qa" : BBHDataset,
    "bbh/logical_deduction_three_objects" : BBHDataset,
    "bbh/logical_deduction_five_objects" : BBHDataset,
    "bbh/logical_deduction_seven_objects" : BBHDataset,
    "bbh/causal_judgement" : BBHDataset,
    "bbh/date_understanding" : BBHDataset,
    "bbh/ruin_names" : BBHDataset,
    "bbh/word_sorting" : BBHDataset,
    "bbh/geometric_shapes" : BBHDataset,
    "bbh/movie_recommendation" : BBHDataset,
    "bbh/salient_translation_error_detection" : BBHDataset,
    "bbh/formal_fallacies" : BBHDataset,
    "bbh/penguins_in_a_table" : BBHDataset,
    "bbh/dyck_languages" : BBHDataset,
    "bbh/multistep_arithmetic_two" : BBHDataset,
    "bbh/navigate" : BBHDataset,
    "bbh/reasoning_about_colored_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_three_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_five_objects" : BBHDataset,
    "bbh/tracking_shuffled_objects_seven_objects" : BBHDataset,
    "bbh/sports_understanding" : BBHDataset,
    "bbh/snarks" : BBHDataset,
    "bbh/web_of_lies" : BBHDataset
}

def extract_multitask_name(task_name: str) -> str:
    return task_name.split('/')[-1]


def create_ds_from_task(task_name: str, **kwargs) -> BaseDataset:
    ds_base = TASK_TO_DS[task_name](**kwargs)

    if isinstance(ds_base, BaseMultiTaskDataset):
        return ds_base.task(extract_multitask_name(task_name))

    return ds_base


def get_task_optimization_metric(ds: BaseDataset) -> str:
    
    if isinstance(ds, BaseClassificationDataset):
        return "f1"
    
    return "meteor"


def get_task_evaluator(ds: BaseDataset):
    # avoid circular imports
    
    from src.evaluation.evaluator import TextClassificationEvaluator, GenerationEvaluator
    
    if isinstance(ds, BaseClassificationDataset):
        return TextClassificationEvaluator()
    
    return GenerationEvaluator()