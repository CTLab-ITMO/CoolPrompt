This is the short description of how our API works.
For better understanding of how our code works from the inside it's recommended to familiarize yourself with these overviews.

---
## `assistant.py`
The main entry point of out framework. It configures all the further processes, selecting an evaluation setup, choosing the optimization method, preparing datasets and etc.

---
## `evaluator/`
- `evaluator.py` - contains our implementation of evaluation workflow
- `metrics.py` - implementation of both classification and generation types of metrics. It supports all of the metrics from **evaluate** library, but you can always create your own metric by overriding the BaseMetric class interface.

---
## `language_model/`
Our tool for easy model loading using VLLM interface. You can provide it with your own configurations to make the interaction with our framework more precise. 

---
## `optimizer/`
Here are implementations of our optimization methods:
- `hype.py` - HyPE method
- `distill_prompt/` - DistillPrompt
- `reflective_prompt/` - ReflectivePrompt. You can see its documentation <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/reflective_prompt/README.md">here</a>.

---
## `utils/`
Foundational utilities. Can be useful if you want to dive deeper in our project.

---