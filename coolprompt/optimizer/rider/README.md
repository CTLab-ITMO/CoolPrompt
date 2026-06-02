# RIDER Genesis Ultra

This package integrates RIDER Genesis Ultra into CoolPrompt as the `rider`
autoprompting method.

## Design

- `core/assistant.py` is the compact `RiderGenesis` facade adapted for the
  CoolPrompt package layout.
- `core/runtime.py`, `core/contract.py`, `core/memory.py`,
  `core/prompt_ops.py`, `core/preservation.py`, `core/run_modes.py`,
  `core/ultra.py`, and `core/synthetic_eval.py` split the RIDER Genesis Ultra
  implementation by responsibility so each module stays reviewable.
- `core/schemas.py` contains the structured Pydantic schemas used by RIDER
  contract extraction, synthetic tests, judge scores, and red-team findings.
- `coolprompt/utils/prompt_templates/rider_templates.py` contains RIDER
  Genesis Ultra meta-prompts following the CoolPrompt prompt-template
  convention.
- `_core_loader.py` loads the copied `RiderGenesis` class and injects the
  LangChain-backed runtime.
- `_llm_shim.py` provides the `rider.llm.client.LLMClient` interface on top of
  LangChain models registered by CoolPrompt.
- `rider.py` exposes the public CoolPrompt wrapper and only allows Ultra mode.

Only the production RIDER Genesis Ultra core is copied into CoolPrompt. The
research repository's benchmark runners, baseline algorithms, dataset loaders,
CLI, evaluation scripts, and experiment templates stay outside this package.
This keeps the build small and keeps prompt text in the standardized template
package.

## Prompt Templates

RIDER Genesis Ultra meta-prompts live in
`coolprompt/utils/prompt_templates/rider_templates.py`.

The active RIDER Genesis Ultra templates include strategy generation,
contract extraction, pairwise comparison, merge, audit, refinement,
synthetic-evaluation, and red-team prompts.

## Usage

```python
from coolprompt.assistant import PromptTuner

prompt_tuner = PromptTuner(target_model=model)
prompt_tuner.run("Improve this prompt", method="rider")
```

Optional role-specific models may be passed through method kwargs as
`planner_model`, `judge_model`, and `critic_model`. If omitted, the target model
is used for all RIDER roles.
