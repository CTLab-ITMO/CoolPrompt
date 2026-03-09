from pathlib import Path
import sys
import time
from prompts import PROMPTS_HUMOR
from langchain_openai.chat_models import ChatOpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))

PROMPTS = PROMPTS_HUMOR
from coolprompt.assistant import PromptTuner  # noqa: 402

# model = DefaultLLM.init(vllm_engine_config={'gpu_memory_utilization':0.99, 'max_model_len':100, 'max_num_seqs':1})
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="",
    temperature=0.7,
    max_tokens=4000,
    timeout=60,
    max_retries=2,
    # rate_limiter=rate_limiter
)
pt = PromptTuner(target_model=model, logs_dir="../test_logs/")

for i, prompt in enumerate(PROMPTS):
    start_run = time.time()

    final_prompt = pt.run(
        start_prompt=prompt,
        task="generation",
        verbose=2,
    )
    print("PROMPT:", final_prompt)
    print("INITIAL METRIC:", pt.init_metric)
    print("FINAL METRIC:", pt.final_metric)
    print("INITIAL PROMPT:", pt.init_prompt)
    print("FINAL PROMPT:", pt.final_prompt)
    print("TOTAL TIME:", time.time() - start_run)
