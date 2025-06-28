<p align="center">
    <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/images/coolprompt_logo.jpg">
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_dark.png">
    <img alt="CoolPrompt Logo" width="80%" height="80%">
    </picture>
</p>

<p>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
    <a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
</p>

<p>
    <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/README.md"><img src="https://img.shields.io/badge/lang-english-red.svg"></a>
    <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/README.ru.md"><img src="https://img.shields.io/badge/lang-russian-gree.svg"></a>
</p>

CoolPrompt - фреймворк для автоматического создания и оптимизации промптов.


##  Установка и запуск

- Установка зависимостей:
```
pip install -r requirements.txt
```
- Импортируем и инициализируем PromptTuner с дефолтной LLM:
```
from coolprompt.assistant import PromptTuner

tuner = PromptTuner()
```
- Или используем свою модель:
```
my_model = VLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    trust_remote_code=True,
    dtype='float16',
)

tuner_with_custom_llm = PromptTuner(model=my_model)
```

## Запуск PromptTuner
- Ассистент поддерживает запуск без датасета
```
# Define an initial prompt
prompt = "Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt = tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```
- Или с ним - в таком случае также будут посчитаны метрики стартового и финального промптов
```
sst2 = load_dataset("sst2")
class_dataset = sst2['train']['sentence']
class_targets = sst2['train']['label']

tuner.run(
    start_prompt=class_start_prompt,
    task="classification",
    dataset=class_dataset,
    target=class_targets,
    metric="accuracy"
)
```
- Промпты и метрики доступны как публичные поля ассистента
```
print("Final prompt:", tuner.final_prompt)
print("Start prompt metric: ", tuner.init_metric)
print("Final prompt metric: ", tuner.final_metric)
```
- Также ассистент работает с задачами генерации

## Больше о проекте
- Исследуйте различные методы авто-промптинга в PromptTuner. Ассистент на данный момент поддерживает HyPE, DistillPrompt, ReflectivePrompt. Вы можете выбрать метод с помощью соответствующего аргумента в `tuner.run`.
- Для ознакомления с фреймворком вы можете увидеть больше примеров в папке `notebooks/` 