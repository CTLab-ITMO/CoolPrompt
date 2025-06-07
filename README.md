<p align="center">
    <img src="https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/docs/images/coolprompt_logo.jpg" alt="logo">
</p>

<p>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
    <a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
    [![Release Notes](https://img.shields.io/github/release/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/releases)
    [![PyPI - License](https://img.shields.io/pypi/l/coolprompt?style=flat-square)](https://opensource.org/licenses/MIT)
    [![PyPI - Downloads](https://img.shields.io/pypi/dm/coolprompt?style=flat-square)](https://pypistats.org/packages/coolprompt)
    [![GitHub star chart](https://img.shields.io/github/stars/CTLab-ITMO/CoolPrompt?style=flat-square)](https://star-history.com/#CTLab-ITMO/CoolPrompt)
    [![Open Issues](https://img.shields.io/github/issues-raw/CTLab-ITMO/CoolPrompt?style=flat-square)](https://github.com/CTLab-ITMO/CoolPrompt/issues)

</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a>Русский</a>
    </p>
</h4>

CoolPrompt is a framework for automative prompting creation.


## Quickstart

- Install requirements with:
```
pip install -r requirements.txt
```
- Import and initialize PromptTuner
```
from coolprompt.assistant import PromptTuner

tuner = PromptTuner()
```
- Run PromptTuner instance with initial prompt:
```
# Define an initial prompt
prompt="Make a summarization of 2+2"

# Run a prompt optimisation
new_prompt=tuner.run(start_prompt=prompt)

# Get your new prompt
print(new_prompt)
```
- Explore more examples in `notebooks/` folder to familiarize yourself with our framework
