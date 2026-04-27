# RIDER Optimizer

RIDER is a multi-strategy prompt optimization method developed in ITMO (CTLab) by D. Dragomirov et al. The integrated `light` mode is the cheapest variant: ~5 LLM calls, ~15s. It runs task analysis → 2 specialized strategies (structural + analytical) → pairwise comparison vs original → quality estimation, guaranteeing the original is replaced only if a strategy beats it.

## Usage

```python
from langchain_core.language_models import BaseLanguageModel

from coolprompt.optimizer.rider import RIDEROptimizer

model: BaseLanguageModel = ...
optimizer = RIDEROptimizer(model)

optimized_prompt = optimizer.optimize("Write a concise product description.")
print(optimized_prompt)
```

## Modes

| Mode | Status |
| --- | --- |
| `light` | ✓ implemented |
| `blitz` | planned |
| `standard` | planned |
| `ultra` | planned |

## Reference

- Master's thesis: Dragomirov D.S., M4137, ITMO 2026.
- RIDER repository: `git@github.com:daglar-dragomirov/rider.git`.

## AutoPromptingMethod Wrapper

An AutoPromptingMethod-style wrapper for the PR #88 interface will be added once PR #88 is merged into `stage`.
