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
[![Telegram Channel](https://img.shields.io/badge/Telegram-2CA5E0?style=flat&logo=telegram&logoColor=white)](https://t.me/+0kMcymeAQrczN2Fi)

CoolPrompt is a framework for automatic prompt creation and optimization.

### Join our [telegram](https://t.me/+0kMcymeAQrczN2Fi) channel to be in touch.

## Practical cases

- Automatic prompt engineering for solving tasks using LLM
- (Semi-)automatic generation of markup for fine-tuning
- Formalization of response quality assessment using LLM
- Prompt adoption for AI Agentic Pipelines
- Etc.

## Core features

- **Optimize prompts** with our APO methods:
    - HyPER / HyPER Light
    - RE-GPS
    - RIDER
    - PromptCompressor
    - *(legacy/deprecated)*: ReflectivePrompt, DistillPrompt
- **LLM-Agnostic Choice:** work with your custom llm (from open-sourced to proprietary) using [supported Langchain LLMs](https://python.langchain.com/docs/integrations/llms/)
- **Develop own custom APO method in one library**
- **Generate synthetic evaluation data** when no input dataset is provided 
- **Evaluate a quality** of prompts incorporating multiple metrics for both classification and generation tasks
- **Evaluate costs** of optimization processes by a number of tokens/calls and a price.
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
cd CoolPrompt

pip install -e .
```

## Quick start

Set your OpenAI API key before running. The default model is `gpt-4o-mini` via the OpenAI API (`OPENAI_API_KEY` environment variable)

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
For technical details and full experimental results, please check our papers + citations inside.

<details close>
    <summary><a href="https://doi.org/10.1145/3803437.3807393"><b>RIDER</b></a></summary>
    
    @inproceedings{dragomirov2026rider,
      author = {Dragomirov, Daglar and Kulin, Nikita and Muravyov, Sergey and Makarov, Ilya and Sukhorukov, Daniil and Mozikov, Mikhail},
      title = {RIDER: Evolutionary Prompt Optimization with Adaptive Operator Selection for Software Engineering},
      booktitle = {Companion Proceedings of the 34th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
      series = {FSE Companion '26},
      year = {2026},
      doi = {10.1145/3803437.3807393}
    }
    
</details>

<details close>
    <summary><a href="https://www.fruct.org/files/publications/volume-38/fruct38/Kul.pdf"><b>CoolPrompt</b></a></summary>
    
    @INPROCEEDINGS{11239071,
      author={Kulin, Nikita and Zhuravlev, Viktor and Khairullin, Artur and Sitkina, Alena and Muravyov, Sergey},
      booktitle={2025 38th Conference of Open Innovations Association (FRUCT)}, 
      title={CoolPrompt: Automatic Prompt Optimization Framework for Large Language Models}, 
      year={2025},
      volume={},
      number={},
      pages={158-166},
      keywords={Technological innovation;Systematics;Large language models;Pipelines;Manuals;Prediction algorithms;Libraries;Prompt engineering;Optimization;Synthetic data},
      doi={10.23919/FRUCT67853.2025.11239071}
    }
    
</details>

<details close>
    <summary><a href="https://ntv.ifmo.ru/file/article/23927.pdf"><b>ReflectivePrompt</b></a></summary>
    
    @misc{zhuravlev2025reflectivepromptreflectiveevolutionautoprompting,
          title={ReflectivePrompt: Reflective evolution in autoprompting algorithms}, 
          author={Viktor N. Zhuravlev and Artur R. Khairullin and Ernest A. Dyagin and Alena N. Sitkina and Nikita I. Kulin},
          year={2025},
          eprint={2508.18870},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2508.18870}, 
    }
    
</details>

<details close>
    <summary><a href="https://arxiv.org/pdf/2508.18992"><b>DistillPrompt</b></a></summary>
    
    @misc{dyagin2025automaticpromptoptimizationprompt,
          title={Automatic Prompt Optimization with Prompt Distillation},
          author={Ernest A. Dyagin and Nikita I. Kulin and Artur R. Khairullin and Viktor N. Zhuravlev and Alena N. Sitkina},
          year={2025},
          eprint={2508.18992},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2508.18992}, 
    }
    
</details>

