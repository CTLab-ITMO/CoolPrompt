from assistant import PromptHelper
import json

prompts = [
    "What is common with the crow and writing table?",
    "А как сделать чтобы langchain создал локальную модель",
    "бро напиши пж решение задачи: доказать что моноид это категория",
]
with open("../data/basic_prompts.json", "r") as f:
    prompts_json = json.load(f)
    for _, prompt in prompts_json.items():
        if isinstance(prompt, str):
            prompts.append(prompt)
        else:
            for _, p in prompt.items():
                prompts.append(p)

ph = PromptHelper()
for prompt in prompts:
    print(f"prompt: {prompt}")
    print(f"result: {ph.invoke(start_prompt=prompt)}")
    print("%" * 80)
