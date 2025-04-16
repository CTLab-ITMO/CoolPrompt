from transformers import AutoTokenizer
from vllm import LLM

def setup_tokenizer(model_name: str) -> AutoTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer # type: ignore


def setup_vllm_model(model_name: str) -> LLM:
    
    model = LLM(model=model_name, dtype="float16", trust_remote_code=True)

    return model # type: ignore