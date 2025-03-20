<p align="center"><h1 align="center">CoolPrompt</h1></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
</p>
<p align="center">
	</p>
<br>


---
## Overview

<overview>
CoolPrompt is an automative prompting framework for automative prompt optimization for Large Language Models (LLMs) and Large Multimodal Models (LMMs)
</overview>

---

## Quick Start

- Install all project requirements
    <code>pip install -r requirements.txt</code>

- Download data
    <code>bash scripts/dataset_downloading.sh --login `minio_login` --password `minio_password` </code>

- Explore example notebooks to familiarize yourself with our framework

---

## Data

CoolPrompt is working with custom NLP benchmark. You can download it from our minio storage using [downloading script](https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/scripts/dataset_downloading.sh) (remember, you need to use credentials to access it).
All data will be downloaded to *~/autoprompting_data*. You can change prompt templates and basic prompts for each dataset manually by editing configuration files there. 
To work with data you need to create custom dataset classes ([SST2Dataset](https://github.com/CTLab-ITMO/CoolPrompt/blob/b088b72de5e9405a720bb4d7157afd9b42ce767f/src/data/classification/sst2_dataset.py#L6) and etc.) using split='train' or 'test' ('test' is by default). You can read about all other parameters in docstrings.
The examples of using benchmark can be found in this [notebook](https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/notebooks/examples/datasets_usage.ipynb).

---

## License

This project is protected under the Apache 2.0 License. For more details, refer to the [LICENSE](https://github.com/CTLab-ITMO/CoolPrompt/LICENSE) file.

---


## Contacts

**WIP**

## Citation

**WIP**

### APA format:

    CTLab-ITMO (2024). CoolPrompt repository [Computer software]. https://github.com/CTLab-ITMO/CoolPrompt

### BibTeX format:

    @misc{CoolPrompt,
        author = {CTLab-ITMO},
        title = {CoolPrompt repository},
        year = {2024},
        publisher = {github.com},
        journal = {github.com repository},
        howpublished = {\url{https://github.com/CTLab-ITMO/CoolPrompt.git}},
        url = {https://github.com/CTLab-ITMO/CoolPrompt.git}
    }

---
