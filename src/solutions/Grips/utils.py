import argparse
import json
import os
import pdb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def setup_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map=device,
                                                quantization_config=quantization_config)
#     model =  LLM(model=model_name, dtype=torch.float16, trust_remote_code=True, \
# quantization="bitsandbytes", load_format="bitsandbytes")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer