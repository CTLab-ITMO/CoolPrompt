import json
import time
import logging
import os
from pathlib import Path
import sys


model_path = "../language_model/models/T-lite-Q8_0.gguf"
provider = "hf_pipeline"
cfg_vllm = {"vllm_engine_config": {"gpu_memory_utilization": 0.95}}
cfg = {}
it = 4

meta_dir = f"logs_providers_{provider}"
os.makedirs(meta_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{meta_dir}/meta_{it}.txt"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def timedif(st):
    return time.time() - st


def logtime(msg, t):
    logger.info(f"{msg}: {t:.4f}s")


sys.path.append(str(Path(__file__).parent.parent.parent))

st = time.time()
from coolprompt.assistant import PromptTuner
from coolprompt.language_model.llm import DefaultLLM

import_time = timedif(st)

logtime("Import time", import_time)

st = time.time()
model = DefaultLLM.init(langchain_provider=provider, **cfg)
model_init_time = timedif(st)
logtime("Model init time", model_init_time)

# st = time.time()
# prompt = "hello! why is so silent there?"
# ans = model.invoke(prompt)
# run_time = timedif(st)
# logtime(f"Prompt '{prompt}' run time", run_time)
# print(ans)

st = time.time()
pt = PromptTuner(model)
pt_init_time = timedif(st)
logtime("PromptTuner init time", pt_init_time)

prompt = "hello! why is so silent there?"
from prompts import PROMPTS_RU as prompts

run_time_log = {}
for i, prompt in enumerate(prompts):
    st = time.time()
    new_prompt = pt.run(prompt, verbose=2)
    run_time = timedif(st)
    logger.info(f"new prompt:\n{new_prompt}\n")
    logtime(f"Prompt '{prompt}' run time", run_time)
    run_time_log[i] = run_time

with open(f"{meta_dir}/meta_{it}.json", "w") as f:
    json.dumps(run_time_log, f)
