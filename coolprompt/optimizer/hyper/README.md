# HyPER Light & HyPER

Meta-prompt–based prompt improvement and the **HyPER** outer loop (paraphrase
candidates, mini-batch scoring, MMR, feedback, inner meta-prompt, validation).

| Module | Role |
|--------|------|
| `meta_prompt.py` | `Optimizer`, `MetaPromptOptimizer`, `HyPERLightMethod` (`hyper_light`). |
| `hyper.py` | `HyPEROptimizer`, `HyPERMethod` (`hyper`), sampling / MMR helpers. |
| `feedback_module.py` | Section-targeted recommendations, contrastive feedback, grouping, leak audit. |
| `coolprompt.utils.prompt_templates.hyper_templates` | Meta-prompt and feedback string templates. |

---

## HyPER Light (`hyper_light`)

One structured meta-prompt and **one** model call. The model should return the new
prompt inside `<result_prompt>…</result_prompt>`.

### Workflow

<p align="center">
    <picture>
    <source srcset="../../../docs/images/hyper_light_pipe.png">
    <img alt="HyPER Light workflow" width="100%" height="100%">
    </picture>
</p>

- **Not data-driven** for optimization: no train/val split is required for the
  optimization step itself (evaluation in `PromptTuner` may still use a dataset).
- **Context**: pass **`hyper_meta_info`** (dict) to `PromptTuner.run`; direct
  method calls use **`meta_info`**. It is merged into the meta-info block. If
  `problem_description` is missing there, it is taken from the
  `problem_description` argument. YAML benchmarks fill the same role via
  `config["meta_info"]` in `run_configured_benchmark`.

### Programmatic

```python
from coolprompt.optimizer.hyper.meta_prompt import HyPERLightMethod

method = HyPERLightMethod()
out = method.optimize(
    model=llm,
    initial_prompt="Your task prompt…",
    problem_description="Short task description",
    hyper_meta_info={"domain": "news"},
)
```

### PromptTuner

```python
from coolprompt.assistant import PromptTuner

pt = PromptTuner(target_llm)
pt.run(
    start_prompt="…",
    task="classification",  # or "generation"
    dataset=…,
    target=…,
    method="hyper_light",
    hyper_meta_info={"problem_description": "…"},
)
```

### Registered name

```python
from coolprompt.utils.var_validation import validate_method

impl = validate_method("hyper_light")
# impl.name == "hyper_light"
```

---

## HyPER (`hyper`)

Several **iterations**: paraphrase around the current best, score candidates on a
train mini-batch, **MMR** (BERTScore diversity), build recommendations (optional
contrastive feedback), optional **instance-leak audit** when
`hyper_meta_info["problem_description"]` is set, then **inner** `MetaPromptOptimizer`
per shortlisted candidate and validation scoring.

### Workflow

<p align="center">
    <picture>
    <source srcset="../../../docs/images/hyper_pipe.png">
    <img alt="HyPER workflow" width="100%" height="100%">
    </picture>
</p>

Typical constructor / YAML **`method`** fields: `n_iterations`, `patience`,
`n_candidates`, `top_n_candidates`, `k_samples`, `mini_batch_size`,
`contrastive_probability`, `enable_instance_leak_audit`, `random_seed`, and truncation
limits for feedback prompts.

Pass **`hyper_meta_info`** the same way as for HyPER Light (includes
`problem_description` for the audit step).

### Programmatic

```python
from coolprompt.optimizer.hyper.hyper import HyPERMethod

method = HyPERMethod()
final = method.optimize(
    model=llm,
    initial_prompt="…",
    dataset_split=(train_x, val_x, train_y, val_y),
    evaluator=evaluator,
    problem_description="…",
    hyper_meta_info={"problem_description": "…"},
    n_iterations=5,
)
```

### PromptTuner

Use `method="hyper"` and supply **dataset** / **target** (data-driven optimizer).

### YAML benchmarks

`coolprompt.method_evaluation.method_evaluation.evaluate_method` with `method="hyper"`
or `"hyper_light"` and a config dict / path; see `method_evaluation` module for the
expected config shape (`dataset`, `meta_info`, nested `method` overrides for HyPER).

---

## Templates

Template literals are in `coolprompt.utils.prompt_templates.hyper_templates`. To
print bundled strings from the repo root:

```bash
python meta_prompts_catalog.py
```
