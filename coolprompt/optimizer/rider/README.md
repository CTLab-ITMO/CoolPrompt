# RIDER Genesis Ultra

This package integrates RIDER Genesis Ultra into CoolPrompt as the `rider`
autoprompting method.

## Design

- `vendor/rider/` is a byte-identical copy of the upstream RIDER package from
  `C:\projects\rider\rider`.
- `_vendor.py` loads the vendored `RiderGenesis` class without editing the
  copied source.
- `_llm_shim.py` provides the `rider.llm.client.LLMClient` interface on top of
  LangChain models registered by CoolPrompt.
- `rider.py` exposes the public CoolPrompt wrapper and only allows Ultra mode.

The vendored tree intentionally keeps RIDER prompts, regex helpers, and internal
pipeline code in their original locations. This keeps the integration auditable:
updates are copied from the upstream RIDER package and verified byte-for-byte by
tests instead of being partially refactored inside CoolPrompt.

## Prompt Templates

RIDER Genesis Ultra prompt templates are included in the vendored source:

- `vendor/rider/assistant.py`: strategy, compare, merge, audit, refine,
  synthetic-evaluation, and red-team meta-prompts used by RiderGenesis Ultra.
- `vendor/rider/algorithms/rider.py`: data-aware initial population seeding
  template for the full RIDER experiment runner.
- `vendor/rider/core/operators.py`: zero-order generation template and
  task-specific fallback prompts.
- `vendor/rider/core/genesis.py`: GENESIS lesson-extraction meta-prompt.

## Usage

```python
from coolprompt.assistant import PromptTuner

prompt_tuner = PromptTuner(target_model=model)
prompt_tuner.run("Improve this prompt", method="rider")
```

Optional role-specific models may be passed through method kwargs as
`planner_model`, `judge_model`, and `critic_model`. If omitted, the target model
is used for all RIDER roles.
