# Spec Generator

`coolprompt.spec_generator` builds a validated task specification from a prompt
and generates synthetic input/output examples for classification and generation
tasks.

Generated examples are structurally validated, duplicate and near-duplicate
inputs are filtered out, and dataset diversity can be guided through
task-specific coverage axes.

## Quick start

```python
from langchain_openai import ChatOpenAI

from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft
from coolprompt.utils.enums import Task

model = ChatOpenAI(model="gpt-4o-mini")
generator = SyntheticDataGenerator(model=model)

result = generator.generate(
    prompt="Classify the emotion expressed in a social-media post.",
    draft=TaskSpecDraft(
        task=Task.CLASSIFICATION,
        labels=("anger", "joy", "optimism", "sadness"),
        output_format="Return exactly one lowercase label.",
    ),
    examples=[
        ("I finally got the job!", "joy"),
        ("I miss my friends.", "sadness"),
    ],
    detect_dataset=False,
    num_samples=40,
    strict_coverage=False,
)

print(result.context.spec)
print(result.examples[0])
print(result.dataset)
print(result.target)
```

`result` is a `GenerationResult` containing:

- `examples`: a tuple of generated `Example` objects;
- `dataset`: generated inputs as a list of strings;
- `target`: generated outputs as a list of strings;
- `context`: the resolved `GenerationContext`, including `context.spec`,
  `context.dataset_name`, and `context.seed_examples`.

Generated examples are returned in memory and are not saved automatically.

## Parameters

### `draft`

`TaskSpecDraft` is an optional partial task specification that can provide any
subset of the fields and constraints described in
[Task specification](#task-specification).

The generator combines the draft with the prompt and available reference
context to build a complete `TaskSpec`.

### `examples`

`examples` accepts either `(input, output)` tuples or `Example` objects:

```python
examples = [
    ("I finally got the job!", "joy"),
    ("I miss my friends.", "sadness"),
]
```

Examples are normalized internally and stored in:

```python
result.context.seed_examples
```

They provide additional context for task-specification inference, distribution
inference, and generation. They are illustrative and are not treated as a
statistical sample for estimating real-world frequencies or target
proportions.

### `detect_dataset`

`False` by default.

When enabled, the generator may detect a supported reference dataset from the
prompt and use it as additional task context. If explicit examples are
provided, they take precedence over examples from the detected dataset.

The resolved dataset name is available as:

```python
result.context.dataset_name
```

### `num_samples`

`40` by default. It must be an integer between `1` and `100`.

The generator returns exactly the requested number of examples or raises
`RuntimeError` if enough valid examples cannot be produced within the bounded
generation process.

### `batch_size`

`15` by default. It controls the maximum generation batch size and must be a
positive integer.

### `task_distribution`

An optional explicit `TaskDistribution`. If omitted while distribution support
is enabled, the generator infers a distribution from the prompt, resolved
`TaskSpec`, and reference examples. See
[Classification label coverage](#classification-label-coverage) for the
behavior of custom distributions in classification tasks.

### `use_task_distribution`

`True` by default.

When enabled, the generator uses either the supplied `task_distribution` or an
inferred distribution. When disabled, no distribution is used.

### `feedback_controlled`

`True` by default.

Enables the coverage-gap loop described in
[Distribution-aware generation](#distribution-aware-generation) and requires
`use_task_distribution=True`.

### `strict_coverage`

`False` by default.

If final coverage remains incomplete:

- `strict_coverage=False` logs the remaining gaps and returns the generated
  dataset;
- `strict_coverage=True` raises `RuntimeError`.

`strict_coverage=True` requires `feedback_controlled=True`. See
[Coverage checks](#coverage-checks) for the semantic limitations of tracked
coverage.

## Task specification

The resolved `TaskSpec` is the generation contract supplied to the model. It
contains:

- task type;
- task description;
- input format;
- output format;
- requirements;
- classification labels;
- language;
- whether an empty output is allowed.

It is available as:

```python
result.context.spec
```

For classification tasks, at least one label is required. For other tasks,
labels must not be provided.

The deterministic validator enforces the example structure, non-empty input,
the empty-output policy, and classification labels. Other semantic properties,
including compliance with `input_format`, `output_format`, `requirements`, and
language, are communicated through the generation prompt and are not
independently verified.

Conceptually:

- `TaskSpec` defines the generation contract;
- `TaskDistribution` defines which dimensions of valid examples should be
  deliberately covered.

## Distribution-aware generation

Distribution-aware, feedback-controlled generation is enabled by default.
When no explicit distribution is supplied, the generator infers meaningful
axes of variation from:

- the original prompt;
- the resolved `TaskSpec`;
- reference examples as additional task context.

During feedback-controlled generation, the generator:

1. tracks model-assigned axis tags for accepted examples;
2. identifies remaining coverage gaps;
3. targets under-covered values in later calls;
4. validates candidates as described in
   [Validation and deduplication](#validation-and-deduplication);
5. may oversample and then select exactly `num_samples` examples while
   preserving coverage as well as possible.

The active distribution and final coverage state are available as:

```python
generator.last_distribution
generator.last_generation_state
```

### Axis strategies

**`BALANCED`**

Targets equal final counts across all values of an axis. If `num_samples`
cannot be divided evenly, target counts differ by at most one.

For three values and ten requested examples, the quotas are `4`, `3`, and `3`.
Actual coverage can remain incomplete when the model does not produce enough
suitable examples.

**`TARGET_PROPORTIONS`**

Targets explicitly specified proportions. Every `AxisValue` must define
`target_ratio`.

The supplied ratios must sum to a value between `0.95` and `1.05`. They are
normalized internally to sum exactly to `1.0` before integer quotas are
allocated.

### Custom distribution

A `TaskDistribution` must contain between one and five uniquely named axes.
Each axis must contain at least two uniquely identified values.

Pass a custom distribution to `SyntheticDataGenerator.generate()` via
`task_distribution`:

```python
from coolprompt.spec_generator.distribution import (
    AxisStrategy,
    AxisValue,
    TaskAxis,
    TaskDistribution,
)

distribution = TaskDistribution(
    axes=(
        TaskAxis(
            name="difficulty",
            description="Difficulty of the generated example.",
            strategy=AxisStrategy.TARGET_PROPORTIONS,
            values=(
                AxisValue(
                    id="easy",
                    description="Straightforward examples.",
                    target_ratio=0.5,
                ),
                AxisValue(
                    id="medium",
                    description="Moderately difficult examples.",
                    target_ratio=0.3,
                ),
                AxisValue(
                    id="hard",
                    description="More difficult examples.",
                    target_ratio=0.2,
                ),
            ),
        ),
    ),
)

result = generator.generate(
    prompt="Generate question-answer pairs.",
    task_distribution=distribution,
    num_samples=40,
)
```

### Classification label coverage

When the generator infers a distribution for a classification task with at
least two labels, it adds a deterministic label axis derived from
`TaskSpec.labels`.

For that axis, the tag of each generated example is derived from its normalized
output. A model-provided label-axis tag is not trusted. For a one-label task,
no separate label axis is added.

A caller-supplied `task_distribution` is used unchanged. If label coverage is
required with a custom distribution, include a compatible label axis
explicitly or rely on the inferred distribution.

### Coverage checks

For non-label axes, the generation model assigns `axis_tags`. The generator:

- normalizes axis names;
- retains only axis/value pairs defined by `TaskDistribution`;
- rejects a candidate during feedback-controlled generation if any required
  distribution axis is missing after validation.

It does not independently verify semantic correctness. A structurally valid
but semantically incorrect model-assigned tag can therefore contribute to the
tracked coverage state. This limitation also applies when
`strict_coverage=True`.

## Validation and deduplication

Generated candidates are validated before acceptance. The deterministic
pipeline rejects examples with:

- invalid `input`/`output` structure;
- empty input;
- empty output when `allow_empty_output=False`;
- classification output outside the allowed label set;
- duplicate input;
- near-duplicate input.

Classification outputs are normalized to the canonical spelling from
`TaskSpec.labels`.

Near-duplicate detection uses normalized input text together with character
and word n-gram cosine similarity. It compares inputs, not outputs or complete
input/output pairs.

Rejected candidates do not count toward the requested sample size. Additional
candidates are requested within a bounded number of top-up attempts. During
feedback-controlled generation, previously accepted examples are also shown
to later generation calls to discourage repetition.

## Generate only a task description

Use `generate_problem_description` when only a concise task description is
needed:

```python
from coolprompt.spec_generator import generate_problem_description
from coolprompt.utils.enums import Task

description = generate_problem_description(
    model=model,
    prompt="Classify the emotion expressed in a social-media post.",
    task=Task.CLASSIFICATION,
    labels=("anger", "joy", "optimism", "sadness"),
    examples=[("I finally got the job!", "joy")],
)
```

`examples` are optional and accept the same tuple or `Example` forms. For
generation tasks, omit `labels`.

## Use synthetic data with HyPER

HyPER is a data-driven prompt optimizer that iteratively proposes prompt
improvements and evaluates them on training and validation examples. Generated
inputs and outputs can be passed to HyPER through `PromptTuner`:

```python
from coolprompt.assistant import PromptTuner

synthetic = generator.generate(
    prompt="Classify the emotion expressed in a social-media post.",
    draft=TaskSpecDraft(
        task=Task.CLASSIFICATION,
        labels=("anger", "joy", "optimism", "sadness"),
    ),
    detect_dataset=False,
    num_samples=40,
)

tuner = PromptTuner(
    target_model=model,
    system_model=model,
)

optimized_prompt = tuner.run(
    start_prompt="Classify the emotion expressed in a social-media post.",
    task="classification",
    dataset=synthetic.dataset,
    target=synthetic.target,
    problem_description=synthetic.context.spec.description,
    method="hyper",
    metric="accuracy",
    validation_size=0.25,
    n_iterations=2,
)
```

Validation on held-out synthetic examples measures performance on data from
the same synthetic-generation process. To evaluate transfer, use a separate
real test set that was not used for task-specification inference, synthetic
generation, prompt optimization, or model selection.
