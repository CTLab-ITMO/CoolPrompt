## BRAVE optimizer

BRAVE is a data-driven evolutionary prompt optimizer. It uses a contextual
controller to choose prompt-transformation operators while respecting a token
budget, and selects the final prompt on a validation split.

Use it through the main CoolPrompt API:

```python
from coolprompt import PromptTuner

tuner = PromptTuner(target_model=model)
prompt = tuner.run(
    start_prompt="Classify the sentiment of the input.",
    task="classification",
    dataset=train_inputs,
    target=train_labels,
    method="brave",
    problem_description="Binary sentiment classification.",
    max_steps=20,
    initial_budget_tokens=50_000,
)
```

BRAVE configuration fields can be passed directly to `PromptTuner.run`, or as
a `BRAVEConfig` instance through the `config` keyword. The low-level
`brave(...)` function, `BRAVEEvoluter`, `BRAVEConfig`, and YAML configuration
loader are exported from `coolprompt.optimizer.brave`.

Set `log_dir` to persist operation logs. Without it, optimization runs without
writing BRAVE-specific log files.
