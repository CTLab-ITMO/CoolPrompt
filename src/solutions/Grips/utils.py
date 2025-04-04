from transformers import AutoTokenizer

def setup_tokenizer(model_name: str) -> AutoTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer # type: ignore