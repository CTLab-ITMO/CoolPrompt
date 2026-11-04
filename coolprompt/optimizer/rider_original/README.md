# RIDER (original)

RIDER — **R**eflective **I**terative **D**iversity-**E**nhanced **R**easoning — is an
evolutionary prompt optimizer. This package implements the version published at
FRUCT: *RIDER: Evolutionary Prompt Optimization with Adaptive Operator Selection and
Reflective Diversity Control* (D. Dragomirov, N. Kulin, ITMO University).

It is registered as `rider_original`. The `rider` method in CoolPrompt is a different,
later system (RIDER Genesis Ultra).

## Algorithm

A population of `N = 12` prompts evolves for `G = 8` generations.

1. **Initialization.** 5 prompts are reverse-engineered from training input–output
   pairs (temperatures 0.8–1.2), 5 are generated from the task description with
   rotating approach templates (0.5–1.3), and 2 fallbacks come from the user's prompt.
   Degenerate outputs (fewer than 15 words or meta-commentary such as
   "Here is the improved prompt:") are regenerated up to three times at a higher
   temperature.
2. **Evaluate** every prompt on the validation split and collect the best prompt's
   failure cases.
3. **Adaptive operator selection.** For each of the `3N` offspring an operator is
   chosen by epsilon-greedy (`ε = 0.10`, `0.25` during stagnation) over Windowed
   Thompson Sampling: a `Beta(α, β)` posterior per operator built from rewards of the
   last 5 generations with decay 0.9, initialized from the operator-class budget
   shares. Operators with mean reward below 0.3 after 10+ trials get the soft penalty
   `α ← 0.3α, β ← max(β, 10)`. The reward of an offspring is 1 if its fitness is
   above the median of the parent population, otherwise 0. The first generation
   has no rewards yet, so its `3N` operators are allocated by the class budget
   shares themselves (largest-remainder rounding).
4. **Generate `3N` offspring** with nine operators, all of which receive the best
   prompt's failure cases (error-directed evolution) and the long-term memory:

   | Class | Operators |
   |---|---|
   | EDA (42%) | `eda_mutation`, `eda_rank_index` |
   | Generative (31%) | `zero_order`, `first_order_refinement` |
   | Evolutionary (20%) | `ga_mutation`, `ga_crossover`, `de_mutation` (DE/current-to-best/1) |
   | Reflective (7%) | `reflection_crossover`, `lamarckian` |

   Reflective operators are disabled for open-ended metrics (BERTScore, ROUGE, BLEU,
   METEOR, LLM-as-judge, G-Eval), where no single correct output exists. Tasks with
   one correct answer, such as GSM8K with exact match, keep them. Pass `open_ended=True` or
   `False` to set this explicitly (for example, SQuAD scored with BERTScore still has one
   correct answer).
5. **Evaluate-first (μ+λ) retention.** All offspring are scored before any decision.
   Parents and offspring are pooled, sorted by fitness and the top `N` kept greedily;
   a prompt with Sentence-BERT cosine similarity above 0.92 to a kept one is skipped
   only if its fitness is within 1%. Diversity breaks ties and never discards a
   clearly better prompt, so the best fitness never decreases.
6. **Stagnation escape.** Improvement below 1% for two generations raises the
   temperature once by 1.3× (capped at 1.5) and ε to 0.25; after four stagnant
   generations the bottom 50% is replaced with fresh zero-order prompts. The
   temperature cools (`T ← 0.9T`) only after genuine improvement from a
   non-stagnated state. The run stops early once a prompt scores the metric's
   maximum (1.0), since nothing can improve on it (`stop_at_perfect=False` disables
   this).
7. **Long-term memory** is refreshed every 3 generations with the top-3 success
   patterns of the population and the bottom-3 failure patterns among all evaluated
   prompts (first sentence plus action verbs).
8. **Output.** The best prompt is returned. A greedy MAP k-DPP (`k = 5`) over quality
   and embedding diversity selects an ensemble, available as
   `method.last_result.ensemble` for majority-vote prediction.

## Usage

### One call

```python
from coolprompt import rider

result = rider(
    model,                                    # any LangChain chat model
    "Classify the emotion of the tweet.",
    texts,
    labels,
    task="classification",
    metric="f1",
    problem_description="Classify tweets as anger, joy, optimism or sadness.",
)

result.best_prompt                # the optimized prompt
print(result.summary())           # scores per generation, operators, best prompt
result.save("rider_result.json")  # everything, as JSON
result.predict(new_texts)         # majority vote of the k-DPP ensemble
```

`rider()` splits the data (stratified, `validation_size=0.25`), scores prompts with the
CoolPrompt metric and prints progress per generation. Pass `system_model=` to let a
different model write the prompts.

### Presets

| Preset | Search | When |
|---|---|---|
| `"paper"` (default) | N=12, G=8, full validation split | results to report |
| `"fast"` | N=8, G=4, fitness on 50 validation examples | quick iteration: ~100 prompts instead of ~300 |
| `"smoke"` | N=4, G=1, 10 validation examples | checking the setup |

```python
result = rider(model, prompt, texts, labels, preset="fast")
result = rider(model, prompt, texts, labels, preset="fast", num_generations=6)
```

Any `RIDERConfig` field can be overridden by name: `population_size`,
`num_generations`, `validation_limit`, `seed`, `max_workers`, `embedding_model`,
`open_ended`, `disabled_operators`, `stop_at_perfect`, and so on.

### Through PromptTuner

```python
from coolprompt import PromptTuner
from coolprompt.optimizer.rider_original import RIDEROriginalMethod

method = RIDEROriginalMethod()
tuner = PromptTuner(target_model=model)
final_prompt = tuner.run(
    start_prompt="Classify the emotion of the tweet.",
    task="classification",
    dataset=texts,
    target=labels,
    method=method,              # or method="rider_original"
    metric="f1",
    system_model_as_optimizer=True,
    rider_preset="fast",        # options work with or without the rider_ prefix
)
print(method.last_result.summary())
```

## Robustness

A model that rejects the `temperature` parameter is called with its defaults. A failed
generation is retried as a degenerate one; a failed evaluation scores the prompt as
`-inf` for that generation instead of stopping the run.

## Cost

With the default configuration RIDER generates about 12 + 8 × 36 = 300 prompts, so the
system model receives roughly 300–400 calls, and every new prompt is evaluated once on
the validation split (evaluations are cached by prompt text).
