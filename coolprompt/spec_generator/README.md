Spec Generator

coolprompt.spec_generator builds a validated task specification from a prompt and generates synthetic input/output examples for classification and generation tasks.

Generation supports validation, deduplication, task-specific coverage axes, and feedback-controlled sampling.

Generate synthetic data

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
print(result.context.spec)
print(result.examples[0])
print(result.dataset)
print(result.target)

examples are input/output pairs used to infer the task specification and guide generation.

Explicit examples take precedence over dataset examples.

With detect_dataset=True, the generator may detect a supported reference dataset and use its examples when explicit examples are not provided.

detect_dataset=False by default.

num_samples must be between 1 and 100. Generation runs in bounded batches and returns exactly the requested number of examples or raises RuntimeError.

Distribution-aware generation

Distribution-aware, feedback-controlled generation is enabled by default.

The generator infers a TaskDistribution describing meaningful task-specific dimensions of variation among valid examples.

A distribution consists of categorical axes such as:

* evidence structure;
* input composition;
* semantic realization;
* temporal or event status;
* input-output relationships;
* other task-specific dimensions of variation.

For classification tasks, the label axis is added deterministically from the task specification.

The inferred axes are used as a coverage scaffold. They are not intended to model the empirical frequency of patterns in the provided examples.

generator.last_distribution
generator.last_generation_state

last_distribution contains the distribution used by the most recent generation run.

last_generation_state contains the final coverage state collected during feedback-controlled generation.

You can also provide an explicit task_distribution using TaskDistribution, TaskAxis, AxisValue, and AxisStrategy.

To disable distribution-aware generation:

result = generator.generate(
    prompt=prompt,
    use_task_distribution=False,
    feedback_controlled=False,
)

feedback_controlled=True requires use_task_distribution=True.

Reference examples

Reference examples are stored in the generation context as context.seed_examples.

They may come from:

* explicit examples passed by the caller;
* a supported dataset selected explicitly;
* a supported dataset detected when detect_dataset=True.

Reference examples guide task inference and generation, but are not assumed to be statistically representative of the source task.

The generator does not infer empirical target proportions from a small reference sample.

Validation

Generated examples are validated before they are accepted.

Invalid, duplicate, and near-duplicate examples are discarded and replacement examples are requested when necessary.

During feedback-controlled generation, candidates must provide one valid axis value for every required distribution axis.

Requested target conditions must also be reflected in the actual generated input/output pair, not only in the reported axis tags.

Previously accepted synthetic examples are supplied to later generation calls to reduce semantic and structural repetition.

If the requested sample count or required coverage cannot be achieved within the configured top-up limits, generation raises RuntimeError.

Task specification

The generated GenerationContext contains the resolved TaskSpec used for synthetic-data generation.

A TaskSpec defines task validity, including properties such as:

* task type;
* task description;
* input format;
* output format;
* requirements;
* classification labels;
* language;
* whether empty output is allowed.

TaskSpec defines what counts as a valid example.

TaskDistribution defines which valid dimensions of variation should be deliberately covered.

Difficulty and reasoning complexity are separate concerns and are not modeled by TaskDistribution.

Generate only a problem description

If you only need a concise task description, use generate_problem_description:

from coolprompt.spec_generator import generate_problem_description
from coolprompt.utils.enums import Task
description = generate_problem_description(
    model=model,
    prompt="Classify the emotion expressed in a social-media post.",
    task=Task.CLASSIFICATION,
    labels=("anger", "joy", "optimism", "sadness"),
    examples=[("I finally got the job!", "joy")],
)

For generation tasks, omit labels.

Examples are optional.

Use synthetic data with HyPER

Generated inputs and outputs can be passed directly to PromptTuner:

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

HyPER validation on synthetic data measures performance on held-out synthetic examples.

To evaluate transfer to real data, use a separate real test set that was not used for:

* task-specification inference;
* synthetic-data generation;
* prompt optimization.

Generated examples are returned in memory and are not saved automatically.