from pathlib import Path
import sys

from langchain_openai.chat_models import ChatOpenAI


proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(proj_root))


from coolprompt.utils.correction.rule import LanguageRule
from coolprompt.test.prompts import PROMPTS_RU


def test_translation(llm):
    for prompt in PROMPTS_RU:
        print(f"################# PROMPT:\n{prompt}\n\n")
        prompt_en = LanguageRule(llm).fix(prompt, {"to_lang": "en"})
        print(f"PROMPT EN:\n{prompt_en}\n\n")
        prompt_ru = LanguageRule(llm).fix(prompt_en, {"to_lang": "ru"})
        print(f"PROMPT RU:\n{prompt_ru}\n\n")


my_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="",
    temperature=0.7,
    max_tokens=4000,
    timeout=60,
    max_retries=2,
    # rate_limiter=rate_limiter
)
# pt = PromptTuner(my_model)
# print(pt.run("ого а что ел Алексей Забашта ?!"))

test_translation(my_model)
