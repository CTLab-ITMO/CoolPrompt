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

### Встроенная LLM
Используется модель t-tech/T-lite-it-1.0 с помощью vLLM:
```
prompt_tuner = PromptTuner()
```

### Кастомизируемая LLM
Используя [поддерживаемые Langchain LLM](https://python.langchain.com/docs/integrations/llms/)

#### [ChatOpenAI](https://python.langchain.com/docs/integrations/llms/openai/)
```
from langchain_openai.chat_models import ChatOpenAI

my_model = ChatOpenAI(
    model="paste model_name"
    base_url="paste paste_url",
    openai_api_key="paste key",
    temperature=0.01,
    max_tokens=500,
)

prompt_tuner = PromptTuner(model=my_model)
```

#### [VLLM](https://python.langchain.com/docs/integrations/llms/vllm/)
```
from langchain_community.llms import VLLM

my_model = VLLM(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    trust_remote_code=True,
    dtype='bfloat16',
)

prompt_tuner = PromptTuner(model=my_model)
```

#### [Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
```
from langchain_ollama.llms import OllamaLLM

# Before run console command `ollama run qwen2.5-coder:32b`

my_model = OllamaLLM(
    model="qwen2.5-coder:32b"
)

prompt_tuner = PromptTuner(model=my_model)
```

#### [Outlines](https://python.langchain.com/docs/integrations/providers/outlines/)
```
from langchain_community.llms import Outlines

my_model = Outlines(
    model="meta-llama/Llama-2-7b-chat-hf",
    backend="transformers", # Backend to use (transformers, llamacpp, vllm, or mlxlm)
    max_tokens=256, # Maximum number of tokens to generate
    streaming=False, # Whether to stream the output
)

prompt_tuner = PromptTuner(model=my_model)
```

#### [HuggingFaceEndpoint](https://python.langchain.com/api_reference/huggingface/llms/langchain_huggingface.llms.huggingface_endpoint.HuggingFaceEndpoint.html)
Hugging Face Hub также предлагает различные эндпоинты для построения ML приложений, используя различные модели с [поддерживаемыми провайдерами](https://huggingface.co/docs/inference-providers/index)
```
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="auto",
    max_new_tokens=100,
    temperature=0.01,
    do_sample=False,
    huggingfacehub_api_token="Your HF-token here"
)
my_model = ChatHuggingFace(llm=llm)

prompt_tuner = PromptTuner(model=my_model)
```

#### [HuggingFacePipeline](https://python.langchain.com/docs/integrations/chat/huggingface/)
Запустить локально, используя HuggingFacePipeline
```
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        temperature=0.01,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
my_model = ChatHuggingFace(llm=llm)

prompt_tuner = PromptTuner(model=my_model)
```

#### [LlamaCpp](https://python.langchain.com/docs/integrations/llms/llamacpp/)
```
from langchain_community.llms import LlamaCpp

my_model = LlamaCpp(
    model_path="Model path",
    temperature=0.01, # Контроль случайности (0-1)
    max_tokens=256, # Максимальное количество токенов в ответе
    top_p=0.1, # Контроль разнообразия
    n_ctx=4096, # Размер контекстного окна
    n_batch=10, # Размер батча для обработки
    verbose=False, # Вывод отладочной информации
)

prompt_tuner = PromptTuner(model=my_model)
```

## Запуск PromptTuner

### Запуск PromptTuner с изначальным промптом
```
# Define an initial prompt
prompt = "Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt = tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```

### Включите датасет для автоматической оптимизации и оценки 
Поданный датасет будет разделен на трейн и тест
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

### Получениe финального промпта и метрик
```
print("Final prompt:", tuner.final_prompt)
print("Start prompt metric: ", tuner.init_metric)
print("Final prompt metric: ", tuner.final_metric)
```

## Больше о проекте
- Исследуйте различные методы авто-промптинга в PromptTuner. CoolPrompt на данный момент поддерживает HyPE, DistillPrompt, ReflectivePrompt. Вы можете выбрать метод с помощью соответствующего аргумента `method` в `tuner.run`.
- Для ознакомления с фреймворком вы можете увидеть больше <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/notebooks/examples">примеров</a> 

## Сотрудничество
- Мы приветствуем и ценим любой вклад и сотрудничество, поэтому вы можете с нами связаться. Для нового кода ознакомьтесь с <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/docs/CONTRIBUTING.md">CONTRIBUTING.md</a>.

## Будущая работа
- Разработка адаптера техников промптинга
- Разработка фичи обратной связи
- И другое