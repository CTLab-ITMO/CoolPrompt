# Migration guide

## Synthetic data generator

The legacy `coolprompt.data_generator` package has been intentionally removed
and replaced by `coolprompt.spec_generator`.

This change introduces a breaking change to the synthetic data generation API.

Before:

```python
from coolprompt.data_generator.generator import SyntheticDataGenerator

dataset, target, description = generator.generate(
    prompt=prompt,
    task=task,
    num_samples=20,
    corner_ratio=0.4,
)
```

After:

```python
from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft

generator = SyntheticDataGenerator(model=model)
result = generator.generate(
    prompt=prompt,
    draft=TaskSpecDraft(task=task),
    num_samples=20,
)

dataset = result.dataset
target = result.target
description = result.context.spec.description
```

`generate()` now returns `GenerationResult`.

The legacy corner-case generation mechanism and the `corner_ratio` option
have been removed. There is no direct replacement for `corner_ratio`.

`TaskDistribution` is a separate feature for controlling generation coverage
and is not equivalent to the former corner-case generation mechanism.

Passing `corner_ratio` to `PromptTuner.run()` raises a migration error instead
of forwarding the unknown argument to an optimizer.