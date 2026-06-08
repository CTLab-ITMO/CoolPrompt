# Contributing to CoolPrompt

Hi there! Thank you for even being interested in contributing to CoolPrompt.
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether they involve new features, improved infrastructure, better documentation, or bug fixes.

## Creating and customizing your own autoprompting method

CoolPrompt methods implement the `AutoPromptingMethod` interface in `coolprompt.optimizer.autoprompting_method`.

Implement three required pieces:

| Member | Purpose |
|---|---|
| `optimize(...)` | Core logic: take `initial_prompt` (+ optional dataset/evaluator) and return an improved prompt |
| `is_data_driven()` | `True` if the method requires `dataset` and `evaluator`; `False` otherwise |
| `name` | Short method id (for logs) |

- `run_configured_benchmark(ctx, start_prompt)` — needed to run YAML-style benchmarks via `method.run()`
- `get_template(task)` — prompt template for evaluation (defaults are provided)

More information and an illustrative example: notebooks/examples/custom_apo_method.ipynb

## Code Standards

We follow next code style standards as [PEP8](https://peps.python.org/pep-0008/) and [Google Style Code](https://google.github.io/styleguide/pyguide.html) in this project. You can install and use basic configuration of Flake8, which combines both practices.

## Docstring Standards

We follow [Google Style Code](https://google.github.io/styleguide/pyguide.html) for formating docstrings. The basic template is below:
```
Description

Args:
    arg1 (dtype): description of arg1
    arg2 (dtype): description of arg2
    ...

Returns:
    return_var1 (dtype): description 
```
