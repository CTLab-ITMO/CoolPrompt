# Re-GPS

## Textual-Gradient AutoPrompting Method

Re-GPS is a data-driven evolutionary optimizer that improves prompts using
failed examples, textual gradients, and mutation. It evaluates prompt
candidates on a train split, gathers low-scoring examples, asks the LLM for
feedback, mutates prompts using that feedback, and validates the best prompt on
the validation split.

### Workflow

<p align="center">
    <picture>
    <source srcset="../../../docs/images/REGPS_pipe.png">
    <img alt="Re-GPS workflow" width="100%" height="100%">
    </picture>
</p>

### Usage

#### Through `PromptTuner`

```python
from coolprompt.assistant import PromptTuner

prompt_tuner = PromptTuner(target_model=model)
final_prompt = prompt_tuner.run(
    start_prompt="Classify the input text.",
    task="classification",
    dataset=train_inputs,
    target=train_targets,
    method="regps",
    problem_description="Classify each input into one of the provided labels.",
    population_size=10,
    num_epochs=5,
)
```

#### Programmatic

```python
from coolprompt.optimizer.regps.run import ReGPSMethod

method = ReGPSMethod()
final_prompt = method.optimize(
    model=model,
    initial_prompt="Classify the input text.",
    dataset_split=(train_x, val_x, train_y, val_y),
    evaluator=evaluator,
    problem_description="Classify each input into one of the provided labels.",
    population_size=10,
    num_epochs=5,
)
```

### Parameters

- `population_size`: number of prompt candidates in each generation.
- `num_epochs`: number of evolutionary iterations.
- `bad_examples_number`: number of low-scoring examples used for feedback.
- `output_path`: directory for Re-GPS outputs and cache files.
- `use_cache`: whether to reuse cached intermediate artifacts.
- `checkpoint_path`: optional path for checkpoint loading/saving.

### Notes

- Re-GPS is data-driven and requires a dataset, targets, and an evaluator.
- If no `problem_description` is provided in benchmark mode, CoolPrompt
  generates one from dataset examples.
