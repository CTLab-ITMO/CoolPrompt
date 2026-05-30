This is a short overview of the public CoolPrompt API and the main internal modules.
For end-to-end usage examples, see `README.md` and `notebooks/examples/`.

---
## `assistant.py`
The main entry point of the framework.

`PromptTuner` configures the target/system language models, detects the task when it is not provided, prepares or generates evaluation data, selects a metric, runs the optimization method, and stores the before/after evaluation results.

Main methods:
- `PromptTuner.run(...)` - optimize a prompt.
- `PromptTuner.test(...)` - run a prompt on a dataset and optionally compute a metric.
- `PromptTuner.get_stats()` / `PromptTuner.reset_stats()` - read/reset usage statistics when the model supports tracking.

The default optimization method is `hyper_light`.

---
## `evaluator/`
- `evaluator.py` - contains our implementation of evaluation workflow
- `metrics.py` - implementation of classification and generation metrics.

Supported metric names:
- classification: `accuracy`, `f1`
- generation: `bleu`, `rouge`, `meteor`, `bertscore`, `codebertscore`, `em`, `geval`, `llm_as_judge`

You can add a custom metric by implementing the `BaseMetric` interface.

---
## `language_model/`
LangChain-compatible model helpers.

CoolPrompt accepts any model that implements LangChain's `BaseLanguageModel` interface. If no model is provided, `PromptTuner` creates the default OpenAI model `gpt-4o-mini` via `ChatOpenAI`, using the `OPENAI_API_KEY` environment variable.

Useful exports:
- `DefaultLLM` - initializes the default OpenAI-backed LangChain model.
- `create_chat_model(...)` - creates a tracked `ChatOpenAI` model.
- `OpenAITracker` / `TrackedLLMWrapper` - collect call, token, and cost statistics for supported OpenAI-compatible models.

---
## `optimizer/`
Implementations of the supported optimization methods.

Method names accepted by `PromptTuner.run(method=...)`:
- `hyper_light` - HyPER Light single-shot meta-prompt optimizer. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/hyper/README.md">Documentation</a>
- `hyper` - iterative HyPER optimizer. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/hyper/README.md">Documentation</a>
- `regps` - RE-GPS optimizer.
- `rider` - RIDER optimizer. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/rider/README.md">Documentation</a>
- `compress` - PromptCompressor. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/prompt_compressor/README.md">Documentation</a>
- `reflective` - legacy ReflectivePrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/reflective_prompt/README.md">Documentation</a>
- `distill` - legacy DistillPrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/distill_prompt/README.md">Documentation</a>

Custom methods should implement `AutoPromptingMethod` and can be passed to `PromptTuner.run(...)` as an instance or a concrete class.

---
## `method_evaluation/`
Benchmark interface for comparing autoprompting methods on dataset/config-based experiments.

`evaluate_method(...)` supports the built-in method names `hyper_light`, `hyper`, `reflective`, `reflectiveprompt`, `distill`, `compress`, `regps`, and `rider`.

---
## `data_generator/` and `task_detector/`
- `data_generator/` - synthetic dataset and target generation when no dataset is provided.
- `task_detector/` - automatic task detection for `classification` and `generation` workflows.

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
