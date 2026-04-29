# SAPO

Segment-based Automatic Prompt Optimization (SAPO) package.

## What is inside

- `strat_pipeline.py`
  - `SAPOPipeline`: primary segment-based optimizer.
  - `AP`: backward-compatible alias to `SAPOPipeline`.
- `pipeline.py`
  - `LegacySAPOPipeline`: legacy candidate-critique-improve loop.
  - `AP`: backward-compatible alias to `LegacySAPOPipeline` in this module.
- `llm.py`
  - `get_openrouter_llm`: OpenRouter client factory with structured output.
  - `get_final_prompt`: helper for 2-message chat payload.
- `schema.py`
  - Pydantic response schemas for structured LLM outputs.
- `prompt.py`
  - Prompt templates used by SAPO pipelines.

## Package exports

From `sapo.__init__`:

- `AP` -> `SAPOPipeline`
- `SAPOPipeline`
- `LegacyAP` -> `LegacySAPOPipeline`
- `LegacySAPOPipeline`

## Dataset format

Both pipelines expect a list of dictionaries with keys:

- `input`: source task input text.
- `reference`: target/reference output text.

Example:

```python
from sapo import SAPOPipeline

dataset = [
    {"input": "Question 1", "reference": "Reference answer 1"},
    {"input": "Question 2", "reference": "Reference answer 2"},
]
```

## Quick start

### Main SAPO pipeline

```python
from sapo import SAPOPipeline

optimizer = SAPOPipeline(
    model_name="anthropic/claude-sonnet-4.5",
    dataset=dataset,
    n_iterations=10,
    early_stopping_rounds=3,
)

best_prompt, best_score = optimizer.run(initial_prompt="Your prompt with {input}")
print(best_score)
```

### Legacy pipeline

```python
from sapo.pipeline import LegacySAPOPipeline

optimizer = LegacySAPOPipeline(
    model_name="anthropic/claude-sonnet-4.5",
    dataset=dataset,
    n_iterations=10,
    n_candidates=5,
    early_stopping_rounds=3,
)

best_prompt, best_score = optimizer.run(initial_prompt="Your prompt with {input}")
```

## Notes

- Prompt templates should contain `{input}` placeholder where task input must be inserted.
- Current evaluation metric is BERTScore F1.
- This README documents current behavior only; no algorithmic upgrades are introduced.

## Migration note

The module folder was renamed:

- old: `new_method`
- new: `sapo`

Update imports accordingly.
