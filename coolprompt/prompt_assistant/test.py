from pathlib import Path
import sys

from langchain_openai import ChatOpenAI

path_proj = str(Path(__file__).resolve().parent.parent.parent)
print(path_proj)
sys.path.append(path_proj)
from coolprompt.assistant import PromptTuner

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="",
    temperature=0.7,
    max_tokens=4000,
    timeout=60,
    max_retries=2,
    # rate_limiter=rate_limiter
)
start_prompt = "а как мне стать лучшей версией себя"
final_prompt = PromptTuner(llm).run(start_prompt)
# assistant = PromptAssistant(llm)
# print(assistant.get_feedback(start_prompt, final_prompt))
