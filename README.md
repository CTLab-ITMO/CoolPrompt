<p align="center">
    <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_dark.png">
    <img alt="CoolPrompt Logo" width="40%" height="40%">
    </picture>
</p>

[![Release Notes](https://img.shields.io/github/release/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/releases)
[![PyPI - License](https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue)](https://opensource.org/license/apache-2-0)
[![PyPI Downloads](https://static.pepy.tech/badge/coolprompt)](https://pepy.tech/projects/coolprompt)
[![GitHub star chart](https://img.shields.io/github/stars/CTLab-ITMO/CoolPrompt?style=flat-square)](https://star-history.com/#CTLab-ITMO/CoolPrompt)
[![Open Issues](https://img.shields.io/github/issues-raw/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/issues)
[![Contributions welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?)](https://github.com/CTLab-ITMO/CoolPrompt/pulls)
[![ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)

CoolPrompt is a framework for automative prompting creation and optimization.

## Practical cases

- Automatic prompt engineering for solving tasks using LLM
- (Semi-)automatic generation of markup for fine-tuning
- Formalization of response quality assessment using LLM
- Prompt tuning for agent systems

## Core features

- **Optimize prompts** with our autoprompting optimizers: HyPE, ReflectivePrompt, DistillPrompt
- **LLM-Agnostic Choice:** work with your custom llm (from open-sourced to proprietary) using [supported Langchain LLMs](https://python.langchain.com/docs/integrations/llms/)
- **Generate synthetic evaluation data** when no input dataset is provided 
- **Evaluate** prompts incorporating multiple metrics for both classification and generation tasks
- **Retrieve feedbacks** to interpret prompt optimization results
- **Automatic task detecting** for scenarios without explicit user-defined task specifications

<p align="center">
    <picture>
    <source srcset="docs/images/coolprompt_scheme.png">
    <img alt="CoolPrompt Scheme" width="100%" height="100%">
    </picture>
</p>

## Quick install
- Install with pip:
```bash
pip install coolprompt
```

- Install with git:
```bash
git clone https://github.com/CTLab-ITMO/CoolPrompt.git

pip install -r requirements.txt
```

## Quick start
Import and initialize PromptTuner using model qwen3-4b-instruct via HuggingFace
```python
from coolprompt.assistant import PromptTuner

prompt_tuner = PromptTuner()

prompt_tuner.run('Write an essay about autumn')

print(prompt_tuner.final_prompt)

# You are an expert writer and seasonal observer tasked with composing a rich,
# well-structured, and vividly descriptive essay on the theme of autumn...
```

## Examples

See more examples in [notebooks](https://github.com/CTLab-ITMO/CoolPrompt/blob/master/notebooks/examples/) to familiarize yourself with our framework


## About project
- The framework is developed by Computer Technologies Lab (CT-Lab) of ITMO University.
- <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/docs/API.md">API Reference</a>

## Contributing
- We welcome and value any contributions and collaborations, so please contact us. For new code check out <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/docs/CONTRIBUTING.md">CONTRIBUTING.md</a>.

## Reference
For technical details and full experimental results, please check our papers.

**CoolPrompt**
```Publishing```

**ReflectivePrompt**
```
@misc{zhuravlev2025reflectivepromptreflectiveevolutionautoprompting,
      title={ReflectivePrompt: Reflective evolution in autoprompting algorithms}, 
      author={Viktor N. Zhuravlev and Artur R. Khairullin and Ernest A. Dyagin and Alena N. Sitkina and Nikita I. Kulin},
      year={2025},
      eprint={2508.18870},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.18870}, 
}
```

**DistillPrompt**
```
@misc{dyagin2025automaticpromptoptimizationprompt,
      title={Automatic Prompt Optimization with Prompt Distillation}, 
      author={Ernest A. Dyagin and Nikita I. Kulin and Artur R. Khairullin and Viktor N. Zhuravlev and Alena N. Sitkina},
      year={2025},
      eprint={2508.18992},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.18992}, 
}
```

