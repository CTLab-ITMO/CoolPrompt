# RIDER Genesis Ultra

This package integrates RIDER Genesis Ultra into CoolPrompt as the `rider`
autoprompting method.

## Design

- `core/assistant.py` is a byte-identical copy of the upstream RIDER Genesis
  Ultra algorithm from `C:\projects\rider\rider\assistant.py`.
- `_core_loader.py` loads the copied `RiderGenesis` class without editing the
  RIDER source.
- `_llm_shim.py` provides the `rider.llm.client.LLMClient` interface on top of
  LangChain models registered by CoolPrompt.
- `rider.py` exposes the public CoolPrompt wrapper and only allows Ultra mode.

Only the production RIDER Genesis Ultra core is copied into CoolPrompt. The
research repository's benchmark runners, baseline algorithms, dataset loaders,
CLI, evaluation scripts, and experiment templates stay outside this package.
This keeps the build small while preserving byte-for-byte auditability for the
algorithm that CoolPrompt executes.

## Prompt Templates

RIDER Genesis Ultra meta-prompts are kept inside the byte-identical
`core/assistant.py` source, because moving them would break source-level parity
with upstream RIDER. CoolPrompt-specific prompt templates remain in
`coolprompt/utils/prompt_templates`.

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
