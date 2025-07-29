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
    English |
    <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/README.ru.md">Русский</a>
</p>

CoolPrompt is a framework for automative prompting creation and optimization.

## Practical cases

- Automatic prompt engineering for solving tasks using LLM
- (Semi-)automatic generation of markup for fine-tuning
- Formalization of response quality assessment using LLM
- Prompt tuning for agent systems

## Quick install
- Install with pip:
```
pip install coolprompt
```

- Install with git:
```
git clone https://github.com/CTLab-ITMO/CoolPrompt.git

pip install -r requirements.txt
```

## Quick start
Import and initialize PromptTuner
```
from coolprompt.assistant import PromptTuner
```

### Default LLM 
Using model t-tech/T-lite-it-1.0 via vLLM interface
```
prompt_tuner = PromptTuner()
```

### Customized LLM 
Using [supported Langchain LLMs](https://python.langchain.com/docs/integrations/llms/)

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
The Hugging Face Hub also offers various endpoints to build ML applications, consisting of different models via [several inference providers](https://huggingface.co/docs/inference-providers/index)
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
Run locally using HuggingFacePipeline
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

## Running PromptTuner

### Run PromptTuner instance with initial prompt
```
# Define an initial prompt
prompt = "Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt = tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```

### Include a dataset for prompt optimization and evaluation 
A provided dataset will be split by trainset and testset
```
sst2 = load_dataset("sst2")
class_dataset = sst2['train']['sentence']
class_targets = sst2['train']['label']

tuner.run(
    start_prompt=prompt,
    task="classification",
    dataset=class_dataset,
    target=class_targets,
    metric="accuracy"
)
```

### Get a final prompt and prompt metrics
```
print("Final prompt:", tuner.final_prompt)
print("Start prompt metric:", tuner.init_metric)
print("Final prompt metric:", tuner.final_metric)
```

## More about project
- Explore the variety of autoprompting methods with PromptTuner: CoolPrompt currently support HyPE, DistillPrompt, ReflectivePrompt. You can choose method via corresponding argument `method` in `tuner.run`
- See more examples in <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/notebooks/examples">notebooks</a> to familiarize yourself with our framework

## Contributing
- We welcome and value any contributions and collaborations, so please contact us. For new code check out <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/docs/CONTRIBUTING.md">CONTRIBUTING.md</a>.

## Future work
- Develop a prompt technique adapter
- Develop a feedback feature
- And more