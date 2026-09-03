# Synthetic Data Generation

Synthetic data generation creates artificial input-output examples for text-based tasks.

It is useful when there is no labeled dataset, when the available dataset is too small, or when extra examples are
needed for testing, validation, prompt evaluation, or model behavior analysis.

The generator can work from a task prompt only. For more controlled and consistent generation, you can also provide an
optional `DataSpec`.

`DataSpec` does not replace the prompt. It gives extra guidance about the task: expected inputs, expected outputs,
labels, constraints, language, and corner cases.

If only a prompt is provided, the generator will build a `TaskSpec` by inferring missing task details from that prompt.
The more explicit the input is, the more controlled and consistent the generated data is likely to be.

---

## Requirements

The generator requires:

- an installed `coolprompt` package;
- a configured language model compatible with LangChain;
- API credentials or local access for the language model you use.

Example with `ChatOpenAI`:

```python
import os
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
)
```

Then pass the model to the generator:

```python
from coolprompt.spec_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(model)
```

---

## Basic Usage

```python
from coolprompt.spec_generator import SyntheticDataGenerator, DataSpec
from coolprompt.utils.enums import Task

generator = SyntheticDataGenerator(model)

result = generator.generate(
    prompt="Generate a synthetic dataset for customer support response rewriting.",
    task=Task.GENERATION,
    user_spec=DataSpec(
        task_description="Rewrite informal customer support replies into polite, professional replies.",
        domain="customer support",
        input_description="An informal or poorly written customer support reply in English.",
        output_description="A polished professional reply with the same meaning.",
        constraints=[
            "Preserve the original meaning.",
            "Do not add new facts.",
            "Use a polite and professional tone.",
            "Return only the rewritten reply.",
        ],
        corner_cases=[
            "Angry or impatient original message",
            "Message with slang or casual abbreviations",
            "Message with unclear wording",
            "Message that is already mostly professional",
        ],
        language="English",
    ),
    examples=[
        (
            "yeah we messed up, send your order number",
            "We made an error. Please send us your order number so we can look into it.",
        ),
        (
            "can't help without more info",
            "Could you please provide a few more details so we can assist you?",
        ),
    ],
    validation=True,
    num_samples=30,
    corner_ratio=0.4,
)
```

---

## Recommended Workflow

Before generating data, it's recommended to review the task specification the generator builds from your inputs. The
`build_spec()` method calls the language model and converts your `prompt`, optional `DataSpec`, and optional `examples`
into a structured `TaskSpec`.

Because `build_spec()` uses a language model, the generated `TaskSpec` may vary slightly across runs. If you want to
keep a specification stable, save it before running `build_spec()` again.

```text
prompt + optional DataSpec + optional examples
        ↓
generator.build_spec(...)
        ↓
TaskSpec
        ↓
optional spec.save(...)       ← save the first generated spec
        ↓
inspect → optionally edit with spec.update()
        ↓
optional spec.save(...)       ← save the approved spec
        ↓
generator.generate(..., spec=spec)
```

### Step 1. Build, inspect, and save the initial spec

```python
from coolprompt.spec_generator import DataSpec
from coolprompt.utils.enums import Task

prompt = "Classify whether an email subject line is professional or unprofessional."

spec = generator.build_spec(
    prompt=prompt,
    user_spec=DataSpec(
        task_description="Classify email subject lines as professional or unprofessional.",
        domain="email communication",
        input_description="A short English email subject line.",
        output_description="Exactly one label: professional or unprofessional.",
        label_set=["professional", "unprofessional"],
        constraints=[
            "Use lowercase labels only.",
            "Do not include explanations.",
        ],
        language="English",
    ),
)

print(spec)
spec.save("specs/email_subject_spec.draft.json")
```

Saving the initial spec is useful because `build_spec()` calls the language model — running it again may produce a
slightly different result.

### Step 2. Update fields that need fixing

```python
spec = spec.update(
    output_description="Exactly one lowercase label: professional or unprofessional.",
    constraints=[
        "Output must be exactly one of: professional, unprofessional.",
        "Use lowercase labels only.",
        "Do not include explanations.",
    ],
)
```

`spec.update()` returns a new `TaskSpec` with only the specified fields changed. Everything else stays as-is.

### Step 3. Save the approved spec

```python
spec.save("specs/email_subject_spec.json")
```

In later runs, load the approved spec instead of calling `build_spec()` again:

```python
from coolprompt.spec_generator.schema import TaskSpec

spec = TaskSpec.load("specs/email_subject_spec.json")
```

Loading a saved spec does not call the language model — it restores the exact `TaskSpec` that was previously saved.

### Step 4. Generate data from the reviewed spec

```python
result = generator.generate(
    prompt=prompt,
    task=Task.CLASSIFICATION,
    spec=spec,
    num_samples=30,
    corner_ratio=0.4,
)
```

When `spec` is passed directly, the generator uses it as-is and skips rebuilding from `prompt`, `user_spec`, or
`examples`.

### Optional: export the spec as editable `DataSpec` code

```python
print(spec.to_data_spec_code())
```

This prints a copy-paste-ready `DataSpec(...)` snippet you can edit and pass back as `user_spec` in future calls.

- Use `spec.save()` and `TaskSpec.load()` when you want reproducible generation from the exact reviewed `TaskSpec`.
- Use `spec.to_data_spec_code()` when you want a human-editable `DataSpec(...)` template.

---

## Working with the Result

`generate()` returns a `GenerationResult` object. The generated data is available directly in memory:

```python
inputs = result.dataset
outputs = result.target
task_description = result.description
task_spec = result.spec
```

The result is not saved automatically. To keep the dataset or the task specification, save them explicitly.

**Convert to a dataframe:**

```python
import pandas as pd

df = pd.DataFrame({
    "input": result.dataset,
    "target": result.target,
})
```

**Save the dataset as CSV:**

```python
df.to_csv("synthetic_data.csv", index=False)
```

**Save the task specification:**

```python
result.spec.save("synthetic_data_spec.json")
```

**Load a saved spec later:**

```python
from coolprompt.spec_generator.schema import TaskSpec

spec = TaskSpec.load("synthetic_data_spec.json")
```

**Export the spec as editable `DataSpec` code:**

```python
print(result.spec.to_data_spec_code())
```

`synthetic_data.csv` contains the generated input-target pairs. `synthetic_data_spec.json` contains the structured
`TaskSpec`: domain, task summary, input format, output format, constraints, labels, corner cases, language, and detected
dataset if any.

---

### Optional Dataset Matching

Dataset matching is disabled by default.

Normally, the generator builds a `TaskSpec` from your `prompt`, optional `DataSpec`, and optional examples. This is the
recommended mode for custom tasks because the generator follows your task description directly instead of applying
benchmark-specific rules.

If you want the generator to use rules for supported benchmark-style tasks, enable dataset matching explicitly:

```python
spec = generator.build_spec(
    prompt=prompt,
    user_spec=user_spec,
    detect_dataset=True,
)
```

## Optional Synthetic Data Specification

The optional specification is passed through `DataSpec`. All fields are optional — fill in only what's relevant to your
task:

```python
DataSpec(
    task_description=None,
    domain=None,
    input_description=None,
    output_description=None,
    label_set=None,
    constraints=None,
    corner_cases=None,
    language=None,
    additional_notes=None,
)
```

### DataSpec Fields

| Field                | What to specify                        |
|----------------------|----------------------------------------|
| `task_description`   | What the model should do.              |
| `domain`             | Task domain or topic area.             |
| `input_description`  | What one input should look like.       |
| `output_description` | What one output should look like.      |
| `label_set`          | Valid labels for classification tasks. |
| `constraints`        | Hard rules every example must follow.  |
| `corner_cases`       | Difficult or unusual cases to include. |
| `language`           | Main language of generated examples.   |
| `additional_notes`   | Extra assumptions or style guidance.   |

---

## Reference Examples

In addition to `DataSpec`, you can pass optional input-output examples through the `examples` argument. The generator
uses them as reference points to understand the desired style, tone, format, and output length.

```python
examples = [
    ("informal input", "polished output"),
    ("another input", "another output"),
]
```

`examples` work together with `DataSpec`: the specification defines the rules, and the examples demonstrate them in
practice.

The examples are used as guidance during generation. They are not automatically included in `result.dataset` or
`result.target`.

---

## Why Use DataSpec

Without `DataSpec`, the generator must infer task details from the prompt. It may come up with something reasonable, but
the output format can drift. For example, a prompt alone might produce:

```text
Professional
This subject line is professional.
formal
not professional
```

With `DataSpec`, the expected behavior is explicit:

```python
DataSpec(
    label_set=["professional", "unprofessional"],
    constraints=[
        "Use lowercase labels only.",
        "Do not include explanations.",
    ],
)
```

And the output becomes consistent:

```text
professional
unprofessional
```

---

## How Fields Affect Generation

### `label_set`

Tells the generator which labels are valid. Without it, the generator may invent labels or use inconsistent wording.

```python
DataSpec(label_set=["positive", "negative", "neutral"])
```

### `constraints`

Hard rules every generated example must follow. Prevents outputs like `The correct label is positive.` or `Positive.`
from slipping through.

```python
DataSpec(
    constraints=[
        "Output must be exactly one label.",
        "Use lowercase labels only.",
        "Do not include explanations.",
    ]
)
```

### `input_description`

Describes what a realistic input looks like. Without this, inputs may be too long, too formal, or off for the task.

```python
DataSpec(input_description="A short English tweet, usually under 280 characters.")
```

### `output_description`

Defines the expected answer format. Especially important for generation tasks where output shape matters.

```python
DataSpec(output_description="Only the final numeric answer. No reasoning, no units.")
```

With this, a math task returns `18` instead of `Samantha has 18 apples.`

### `corner_cases`

Asks the generator to include tricky or unusual examples — not just the easy, textbook cases.

```python
DataSpec(
    corner_cases=[
        "Very short inputs",
        "Inputs with informal language",
        "Ambiguous wording",
    ]
)
```

### `additional_notes`

A place for anything important that doesn't fit the other fields.

```python
DataSpec(
    additional_notes=(
        "Assume a formal corporate workplace. Emojis, slang, and excessive punctuation "
        "should be treated as unprofessional."
    )
)
```

---

## Before You Generate

For the most consistent results, provide at least:

- `task_description`
- `input_description`
- `output_description`
- `label_set` (for classification tasks)
- `constraints`
- `language`
- a few `examples`, when output style, tone, or format matters

The more context you give the generator, the less it has to guess — and the more reliable your data will be.

---

## Weak vs. Strong Specification

**Weak — minimal input:**

```python
result = generator.generate(
    prompt="Classify email subject lines.",
    task=Task.CLASSIFICATION,
    num_samples=10,
)
```

The generator has to figure out on its own: which labels to use, what a valid input looks like, what format the output
should be in, and whether explanations are allowed. The data may still be usable — just less predictable.

**Strong — fully specified:**

```python
result = generator.generate(
    prompt="Classify whether an email subject line is professional or unprofessional.",
    task=Task.CLASSIFICATION,
    user_spec=DataSpec(
        task_description="Classify email subject lines as professional or unprofessional.",
        domain="email communication",
        input_description="A short English email subject line.",
        output_description="Exactly one label: professional or unprofessional.",
        label_set=["professional", "unprofessional"],
        constraints=[
            "Output must be exactly one of: professional, unprofessional.",
            "Use lowercase labels only.",
            "Do not include explanations.",
        ],
        corner_cases=[
            "Subject lines with emojis",
            "Very informal subject lines",
            "Overly long subject lines",
            "Polite but vague subject lines",
        ],
        additional_notes="Assume a formal corporate workplace.",
        language="English",
    ),
    num_samples=10,
    corner_ratio=0.4,
)
```

The generator has clear rules to work with, and the output is much more consistent.

---

## Regular and Corner-Case Examples

The generator produces two kinds of examples: regular ones and corner cases.

`corner_ratio` controls the balance — it's a float between `0.0` and `1.0`, with a default of `0.4`.

```python
num_samples = 10
corner_ratio = 0.4
# → 6 regular examples, 4 corner-case examples
```

If you don't specify any corner cases, the generator will infer them from your task specification.

---

## Validation

Enable validation by passing `validation=True`:

```python
result = generator.generate(
    prompt=prompt,
    task=Task.CLASSIFICATION,
    spec=spec,
    num_samples=30,
    corner_ratio=0.4,
    validation=True,
)
```

When validation is enabled, examples pass through four stages:

1. Format validation — checks required fields, value types, labels, and task-specific rules.
2. Duplicate filtering — removes exact and near-duplicate inputs across the full run.
3. LLM judge — checks semantic correctness and compliance with the TaskSpec.
4. Top-up generation — generates replacements for rejected examples until the target size or attempt limit is reached.

Regular and corner-case examples are validated separately but share the same duplicate-detection state.

## Dataset-Specific Rules

Dataset-specific rules are disabled by default.

To enable matching for supported benchmark-style tasks, pass `detect_dataset=True` when building the specification:

Currently supported:

| Dataset      | Task                            |
|--------------|---------------------------------|
| `tweeteval`  | Tweet emotion classification    |
| `gsm8k`      | Grade-school math reasoning     |
| `common_gen` | Concept-to-sentence generation  |
| `squad_v2`   | Context question answering      |
| `xsum`       | One-sentence news summarization |

If the task doesn't match any of these, the generator falls back to generic templates.