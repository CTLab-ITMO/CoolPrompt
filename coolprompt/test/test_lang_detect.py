from pathlib import Path
import sys
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai.chat_models import ChatOpenAI


proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(proj_root))

from prompts import PROMPTS_EN, PROMPTS_RU
from coolprompt.utils.language_detection import detect_language


def test_prompt(prompts, target, test_func):
    errors = []
    for prompt in prompts:
        lang = test_func(prompt)
        if lang != target:
            print(
                f"###### Error for prompt:\n{prompt}\n{{lang is {lang}}}\n\n"
            )
            errors.append((prompt, lang, target))
    return errors


def test_prompts(test_func):
    errors = []
    errors.extend(test_prompt(PROMPTS_RU, "ru", test_func))
    errors.extend(test_prompt(PROMPTS_EN, "en", test_func))
    print(f"errors: {len(errors)}")
    for prompt, lang, target in errors:
        print(f"ERROR:\nprompt:\n{prompt}\nlang={lang}, target={target}\n\n")


def test_detect():
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # 1 запрос в секунду
        check_every_n_seconds=0.1,  # проверять каждые 100ms
        max_bucket_size=10,  # максимальный размер буфера
    )

    my_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="",
        temperature=0.7,
        max_tokens=4000,
        timeout=60,
        max_retries=2,
        # rate_limiter=rate_limiter
    )

    test_prompts(lambda x: detect_language(x, my_model))


test_detect()
