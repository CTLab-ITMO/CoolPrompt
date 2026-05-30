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
- `meta_prompt.py` - HyPER Light (single-shot :class:`~coolprompt.optimizer.hyper.meta_prompt.MetaPromptOptimizer`). <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/hyper/README.md">Documentation</a>
- `distill_prompt/` - DistillPrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/distill_prompt/README.md">Documentation</a>
- `reflective_prompt/` - ReflectivePrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/reflective_prompt/README.md">Documentation</a>

---
## `utils/`
Foundational utilities. Can be useful if you want to dive deeper in our project.

---

### Structured LLM output

Every LLM call in CoolPrompt can be routed through LangChain's
`with_structured_output(schema, method="json_schema")` instead of free-form
text generation, so the answer is returned as a populated pydantic model.
The schemas live under [`coolprompt.utils.structured_schemas`](../coolprompt/utils/structured_schemas/__init__.py:1)
(one sub-package per consumer: `data_generator/`, `evaluator/`, `hyper/`,
`reflective_prompt/`, `regps/`, `language_model/`).

The feature is opt-in per consumer via a `use_structured_output: bool = False`
flag:

* Optimizers — [`RegpsEvoluter`](../coolprompt/optimizer/regps/evoluter.py:1),
  [`ReflectiveEvoluter`](../coolprompt/optimizer/reflective_prompt/evoluter.py:1),
  [`Hyper`](../coolprompt/optimizer/hyper/hyper.py:1) /
  [`MetaPrompt`](../coolprompt/optimizer/hyper/meta_prompt.py:1) /
  [`FeedbackModule`](../coolprompt/optimizer/hyper/feedback_module.py:1),
  [`PromptCompressor`](../coolprompt/optimizer/prompt_compressor/compressor.py:1).
* Evaluation — [`Evaluator`](../coolprompt/evaluator/evaluator.py:1),
  [`JudgeMetric`](../coolprompt/evaluator/metrics.py:416),
  [`GEvalMetric`](../coolprompt/evaluator/metrics.py:507) (forwards the flag
  to the underlying [`DeepEvalLangChainModel`](../coolprompt/language_model/deepeval_model.py:13),
  which then honours the pydantic `schema` DeepEval passes to its judges;
  when no schema is supplied the wrapper falls back to
  [`DeepEvalJudgeResponse`](../coolprompt/utils/structured_schemas/language_model/schemas.py:4)).
* Pre-processing — [`DataGenerator`](../coolprompt/data_generator/generator.py:1),
  [`TaskDetector`](../coolprompt/task_detector/detector.py:1).

The shared transport layer ([`TrackedLLMWrapper`](../coolprompt/language_model/tracker.py:79)
and [`DefaultLLM`](../coolprompt/language_model/llm.py:12)) is intentionally
schema-agnostic — callers decide the schema at call site through
`model.with_structured_output(...)`.


---

