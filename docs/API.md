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

The default optimization method is `hyper_light`. `method="auto"` enables the
metadata-driven APO selector. It uses the system model to profile the task and
retrieve similar datasets, then ranks comparable configurations by quality,
cost, and runtime. The result is stored in `PromptTuner.meta_selection` and
included in telemetry exports. `meta_classifier_path` overrides the bundled
CSV; `meta_dataset_name` is an optional retrieval hint. SAPO recommendations
fall back to `hyper`, because SAPO is not yet implemented in CoolPrompt.

---
## `evaluator/`
- `evaluator.py` - contains our implementation of evaluation workflow
- `metrics.py` - implementation of classification and generation metrics.

Supported metric names:
- classification: `accuracy`, `f1`
- generation: `bleu`, `rouge`, `meteor`, `bertscore`, `multiref_bertscore`, `codebertscore`, `em`, `geval`, `llm_as_judge`

`multiref_bertscore` is intended for CommonGen-style multi-reference
generation tasks. For each prediction it computes BERTScore F1 against every
reference and uses the maximum reference score; the final score is the mean
over examples.

For BERTScore-based metrics, the default underlying model is
`bert-base-multilingual-cased`. You can override it with
`bertscore_model_type`; the same model type is used for both `bertscore`
and `multiref_bertscore`.

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
- `auto` - metadata-driven selection between supported data-driven methods.
- `regps` - RE-GPS optimizer.
- `rider` - RIDER optimizer. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/rider/README.md">Documentation</a>
- `brave` - BRAVE budget-aware evolutionary optimizer. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/brave/README.md">Documentation</a>
- `compress` - PromptCompressor. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/prompt_compressor/README.md">Documentation</a>
- `reflective` - legacy ReflectivePrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/reflective_prompt/README.md">Documentation</a>
- `distill` - legacy DistillPrompt. <a href="https://github.com/CTLab-ITMO/CoolPrompt/blob/master/coolprompt/optimizer/distill_prompt/README.md">Documentation</a>

Custom methods should implement `AutoPromptingMethod` and can be passed to `PromptTuner.run(...)` as an instance or a concrete class.

---
## `method_evaluation/`
Benchmark interface for comparing autoprompting methods on dataset/config-based experiments.

`evaluate_method(...)` supports the built-in method names `hyper_light`, `hyper`, `reflective`, `reflectiveprompt`, `distill`, `compress`, `regps`, `rider`, and `brave`.

---

## `spec_generator/` and `task_detector/`

- [`spec_generator/`](../coolprompt/spec_generator/README.md) — task
  specification inference and controlled synthetic-data generation with
  validation, deduplication, and distribution-aware coverage.
- `task_detector/` — automatic task detection for `classification` and
  `generation` workflows.

For usage examples, configuration options, coverage behavior, and validation
limitations, see the
[`spec_generator` documentation](../coolprompt/spec_generator/README.md).

The legacy `coolprompt.data_generator` package has been removed and replaced
by `coolprompt.spec_generator`. This is a breaking API change. See
[`MIGRATION.md`](MIGRATION.md) for migration instructions.

---
## `utils/`
Foundational utilities. Can be useful if you want to dive deeper in our project.


---
