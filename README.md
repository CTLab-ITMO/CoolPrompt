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

- with default LLM
```
prompt_tuner = PromptTuner()
```

- or __customize your own LLM__ using supported Langchain LLMs
- List of available LLMs: https://python.langchain.com/docs/integrations/llms/
```
from langchain_community.llms import VLLM

my_model = VLLM(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    trust_remote_code=True,
    dtype='bfloat16',
)

prompt_tuner = PromptTuner(model=my_model)
```

## Running PromptTuner
- Run PromptTuner instance with initial prompt
```
# Define an initial prompt
prompt = "Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt = tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```

- including a dataset for prompt optimization and evaluation. 
A provided dataset will be split by trainset and testset.
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

- to get a final prompt and prompt metrics
```
print("Final prompt:", tuner.final_prompt)
print("Start prompt metric:", tuner.init_metric)
print("Final prompt metric:", tuner.final_metric)
```
- This also works for generation tasks

## More about project
- Explore the variety of autoprompting methods with PromptTuner: CoolPrompt currently support HyPE, DistillPrompt, ReflectivePrompt. You can choose method via corresponding argument `method` in `tuner.run`
- See more examples in <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/notebooks/examples">notebooks</a> to familiarize yourself with our framework
