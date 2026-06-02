# RIDER Genesis Ultra

This package integrates RIDER Genesis Ultra into CoolPrompt as the `rider`
autoprompting method.

## Design

- `core/assistant.py` contains the RIDER Genesis Ultra algorithm adapted for
  the CoolPrompt package layout.
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
