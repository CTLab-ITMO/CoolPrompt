<p align="center">
    <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_dark.png">
    <img alt="CoolPrompt Logo" width="40%" height="40%">
    </picture>
</p>

[![Release Notes](https://img.shields.io/github/release/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/releases)
[![PyPI - License](https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue)](https://opensource.org/license/apache-2-0)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/coolprompt?style=flat-square)](https://pypistats.org/packages/coolprompt)
[![GitHub star chart](https://img.shields.io/github/stars/CTLab-ITMO/CoolPrompt?style=flat-square)](https://star-history.com/#CTLab-ITMO/CoolPrompt)
[![Open Issues](https://img.shields.io/github/issues-raw/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/issues)
[![ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)


<p align="center">
    <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/README.md">English</a> | 
    Русский
</p>

CoolPrompt - фреймворк для автоматического создания и оптимизации промптов.

## Практическое применение

- Автоматическое создание промптов для решения задач с использованием LLM
- (Полу-)автоматическая генерация разметки для файнтюнинга
- Формализация оценки качества ответов с использованием LLM
- Тюнинг инструкций в агентных системах

## Установка
- Установка через pip:
```
pip install coolprompt
```

- Установка через git:
```
git clone https://github.com/CTLab-ITMO/CoolPrompt.git

pip install -r requirements.txt
```

## Быстрый запуск

Импортируем и инициализируем PromptTuner
```
from coolprompt.assistant import PromptTuner
```

- с встроенной LLM:
```
prompt_tuner = PromptTuner()
```

- или __кастомизируем свою модель__ с помощью поддерживаемых Langchain LLM
- Список поддерживаемых LLM: https://python.langchain.com/docs/integrations/llms/
```
from langchain_community.llms import VLLM

my_model = VLLM(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    trust_remote_code=True,
    dtype='float16',
)

prompt_tuner = PromptTuner(model=my_model)
```

## Запуск PromptTuner
- Запуск PromptTuner с изначальным промптом
```
# Define an initial prompt
prompt = "Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt = tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```

- Или включив датасет для автоматической оптимизации и оценки. Поданный датасет будет разделен на трейн и тест.
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

- Для получения финального промпта и метрик
```
print("Final prompt:", tuner.final_prompt)
print("Start prompt metric: ", tuner.init_metric)
print("Final prompt metric: ", tuner.final_metric)
```
- Также ассистент работает с задачами генерации

## Больше о проекте
- Исследуйте различные методы авто-промптинга в PromptTuner. CoolPrompt на данный момент поддерживает HyPE, DistillPrompt, ReflectivePrompt. Вы можете выбрать метод с помощью соответствующего аргумента `method` в `tuner.run`.
- Для ознакомления с фреймворком вы можете увидеть больше <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/notebooks/examples">примеров</a> 

## Сотрудничество
- Мы приветствуем и ценим любой вклад и сотрудничество, поэтому вы можете с нами связаться. Для нового кода ознакомьтесь с <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/docs/CONTRIBUTING.md">CONTRIBUTING.md</a>.