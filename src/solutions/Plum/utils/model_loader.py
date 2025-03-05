import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "AnatoliiPotapov/T-lite-instruct-0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device_map = "auto"
else:
    device_map = None
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
if device_map is None:
    model = model.to(device)
model.eval()
