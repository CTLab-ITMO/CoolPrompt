# Spec Generator

`coolprompt.spec_generator` turns a task prompt into a validated `TaskSpec` and generates input/output examples for classification or generation tasks. It can infer coverage axes, track which values have been generated, and request examples for gaps in later batches.

## Generate a dataset

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
)

print(result.context.spec)  # Inferred specification with explicit draft overrides
print(result.examples[0])   # Example(input=..., output=...)
print(result.dataset)       # list[str]: generated inputs
print(result.target)        # list[str]: generated outputs
```

`examples` are trusted input/output pairs used to infer the specification and guide generation. Pass `distribution_examples` when the examples used to infer coverage should differ from these seed examples. With `detect_dataset=True` (the default), the generator may identify one of the supported reference datasets and use its examples when no explicit examples are supplied. Explicit examples take precedence.

The generator requests exactly `num_samples` examples, in batches of at most `batch_size`. `num_samples` must be between 1 and 100; the defaults are 40 and 15, respectively. If validation cannot obtain enough acceptable examples within the top-up limit, generation raises `RuntimeError`.

## Coverage and validation

Distribution-aware, feedback-controlled generation is enabled by default. The generator infers a `TaskDistribution` from the task and reference examples, records accepted axis values, and targets underrepresented values in later batches. To control coverage yourself, pass a validated `task_distribution` built from `TaskDistribution`, `TaskAxis`, `AxisValue`, and `AxisStrategy` in `coolprompt.spec_generator.distribution`.

```python
generator.last_distribution      # TaskDistribution | None
generator.last_generation_state  # GenerationState | None
```

`last_generation_state` is populated only for feedback-controlled runs. Set `use_task_distribution=False, feedback_controlled=False` to generate without axes. `feedback_controlled=True` requires `use_task_distribution=True`.

Feedback-controlled runs validate examples, reject invalid or duplicate candidates, and request replacements. `structural_validation=True` additionally enables semantic and structural repetition filtering. Outside feedback-controlled mode, the validation pipeline runs only when `structural_validation=True`.

## Generate only a problem description

If you already have a dataset and need a short description for an optimizer, use the standalone helper. It requests only a description, rather than a complete `TaskSpec`:

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

For generation tasks, omit `labels`. Examples are optional; the task prompt remains the primary source of requirements.

## Synthetic data with HyPER

`PromptTuner` accepts the generated inputs and targets directly. Pass the inferred description so the tuner does not make another description-generation call.

```python
from langchain_openai import ChatOpenAI

from coolprompt.assistant import PromptTuner
from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft
from coolprompt.utils.enums import Task

initial_prompt = "Classify the emotion expressed in a social-media post."
model = ChatOpenAI(model="gpt-4o-mini")
generator = SyntheticDataGenerator(model=model)

synthetic = generator.generate(
    prompt=initial_prompt,
    draft=TaskSpecDraft(
        task=Task.CLASSIFICATION,
        labels=("anger", "joy", "optimism", "sadness"),
        output_format="Return exactly one lowercase label.",
    ),
    examples=[
        ("I finally got the job!", "joy"),
        ("I miss my friends.", "sadness"),
        ("I am furious that my work was deleted.", "anger"),
        ("Tomorrow gives me another chance.", "optimism"),
    ],
    detect_dataset=False,
    num_samples=40,
)

tuner = PromptTuner(target_model=model, system_model=model)
optimized_prompt = tuner.run(
    start_prompt=initial_prompt,
    task="classification",
    dataset=synthetic.dataset,
    target=synthetic.target,
    problem_description=synthetic.context.spec.description,
    method="hyper",
    metric="accuracy",
    validation_size=0.25,
    n_iterations=2,
)

print("Initial validation score:", tuner.init_metric)
print("Final validation score:", tuner.final_metric)
print("Optimized prompt:", optimized_prompt)
```

In this example, HyPER's validation score is measured on held-out **synthetic** examples. To measure transfer to real data, evaluate the initial and optimized prompts on the same independent real test set. Keep that test set out of specification inference, generation, and optimization.

Generated results are returned in memory and are not saved automatically.
