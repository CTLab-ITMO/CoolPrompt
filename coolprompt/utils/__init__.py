import os
from random import random

import numpy as np
import torch


DEFAULT_MODEL_NAME = "t-tech/T-lite-it-1.0"
DEFAULT_MODEL_PARAMETERS = {
    "max_new_tokens": 4000,
    "temperature": 0.5,
}
NAIVE_AUTOPROMPTING_PROMPT_TEMPLATE = (
    "Rewrite the following prompt to maximize its effectiveness for LLMs.\n"
    "Apply transformations: structure, specifics, remove ambiguity, add example, keep intent.\n"
    "Only output the rewritten prompt, with no explanation or formatting.\n"
    "\n"
    "Prompt:\n"
    "<PROMPT>\n"
    "Rewritten prompt:\n"
)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)