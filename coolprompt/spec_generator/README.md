# Spec Generator

`coolprompt.spec_generator` builds a task specification and generates synthetic datasets for `classification` and `generation` tasks.

```text
prompt + examples + optional TaskSpecDraft
                    ↓
                SpecBuilder
                    ↓
             GenerationContext
             ├── TaskSpec
             ├── dataset_name
             └── seed_examples
                    ↓
         optional TaskDistribution
                    ↓
                generation
                    ↓
     optional validation + deduplication
                    ↓
             GenerationResult
```

## Quick start

```python
from coolprompt.spec_generator import Example, SyntheticDataGenerator, TaskSpecDraft
from coolprompt.utils.enums import Task

result = SyntheticDataGenerator(model).generate(
    prompt="Classify the emotion in a social-media post.",
    draft=TaskSpecDraft(
        task=Task.CLASSIFICATION,
        labels=("anger", "joy", "optimism", "sadness"),
        output_format="Return exactly one lowercase label.",
    ),
    examples=(
        Example(input="I finally got the job!! 🎉", output="joy"),
        Example(input="Tomorrow is another chance.", output="optimism"),
        Example(input="Why did the app delete my work AGAIN?", output="anger"),
        Example(input="I miss how things used to be.", output="sadness"),
    ),
    num_samples=100,
    batch_size=10,
)
```

## Full example: synthetic generation + HyPER

This example generates 100 synthetic samples, optimizes the initial prompt with `hyper`, and saves the main artifacts.

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from coolprompt.assistant import PromptTuner
from coolprompt.spec_generator import Example, SyntheticDataGenerator, TaskSpecDraft
from coolprompt.utils.enums import Task

load_dotenv()

INITIAL_PROMPT = """
Classify the dominant emotion in the input.
Return exactly one label: anger, joy, optimism, or sadness.
""".strip()

system_model = ChatOpenAI(
    model=os.getenv("SYSTEM_MODEL", "gpt-4o-mini"),
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
)
target_model = ChatOpenAI(
    model=os.getenv("TARGET_MODEL", "gpt-4o-mini"),
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)

examples = (
    Example(input="@user I finally got the job!! 🎉 #happy", output="joy"),
    Example(input="Today was rough, but tomorrow gives us another chance.", output="optimism"),
    Example(input="The app deleted my draft AGAIN. Absolutely furious.", output="anger"),
    Example(input="I honestly feel empty and miss everyone.", output="sadness"),
)

generator = SyntheticDataGenerator(model=system_model, task_spec_model=system_model)
synthetic = generator.generate(
    prompt=INITIAL_PROMPT,
    dataset_name="tweeteval",
    draft=TaskSpecDraft(
        task=Task.CLASSIFICATION,
        description="Classify the dominant emotion in a short social-media post.",
        input_format="One short English social-media post.",
        output_format="Exactly one lowercase label.",
        requirements=("Return no explanation.",),
        labels=("anger", "joy", "optimism", "sadness"),
        language="English",
    ),
    examples=examples,
    distribution_examples=examples,
    detect_dataset=False,
    num_samples=100,
    batch_size=10,
    use_task_distribution=True,
    feedback_controlled=True,
    structural_validation=True,
)

tuner = PromptTuner(
    target_model=target_model,
    system_model=system_model,
    logs_dir="run_logs/hyper",
)
optimized_prompt = tuner.run(
    start_prompt=INITIAL_PROMPT,
    task="classification",
    dataset=synthetic.dataset,
    target=synthetic.target,
    method="hyper",
    metric="f1",
    problem_description=synthetic.context.spec.description,
    validation_size=0.2,
    batch_size=20,
    hyper_meta_info={
        "input_format": synthetic.context.spec.input_format,
        "output_format": synthetic.context.spec.output_format,
        "requirements": synthetic.context.spec.requirements,
    },
    system_model_as_optimizer=True,
    n_iterations=3,
    patience=2,
    n_candidates=3,
    top_n_candidates=2,
    k_samples=3,
    mini_batch_size=16,
    random_seed=42,
)

output_dir = Path("results/tweeteval_hyper")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")
(output_dir / "synthetic_data.json").write_text(
    json.dumps(synthetic.model_dump(mode="json"), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
if generator.last_distribution is not None:
    (output_dir / "task_distribution.json").write_text(
        generator.last_distribution.model_dump_json(indent=2),
        encoding="utf-8",
    )

print("Initial score:", tuner.init_metric)
print("Final score:", tuner.final_metric)
print("Optimized prompt:\n", optimized_prompt)
```

HyPER splits the synthetic dataset into training and validation subsets. Evaluate final quality separately on a fixed real-world test set that was not used for generation or optimization.

## Main parameters

| Parameter | Purpose |
|---|---|
| `draft` | Explicit overrides for the inferred `TaskSpec` |
| `examples` | Trusted examples used for specification and generation |
| `distribution_examples` | Reference examples used to infer axes and guide feedback-controlled generation |
| `task_distribution` | Prebuilt `TaskDistribution` used instead of inference |
| `detect_dataset` | Automatically detect a supported dataset |
| `use_task_distribution` | Generate with distribution-aware guidance |
| `feedback_controlled` | Target underrepresented axis values in later batches |
| `structural_validation` | Filter semantic and structural repetitions |

`feedback_controlled=True` requires `use_task_distribution=True`.
The validation pipeline always runs in feedback-controlled mode. Otherwise, it runs only when `structural_validation=True`.

Supported datasets: `common_gen`, `gsm8k`, `squad_v2`, `tweeteval`, and `xsum`.

## Result

```python
result.examples   # tuple[Example, ...]
result.dataset    # list[str] — generated inputs
result.target     # list[str] — generated outputs
result.context    # GenerationContext

generator.last_distribution       # TaskDistribution | None
generator.last_generation_state   # GenerationState | None
```

Results are not saved automatically. The maximum `num_samples` value is 100. The pipeline does not use a separate corner-case generation phase.
