import json

import pytest

from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft
from coolprompt.spec_generator.distribution import (
    AxisValue,
    GenerationState,
    TaskAxis,
    TaskDistribution,
    coverage_gaps,
)
from coolprompt.spec_generator.utils.retry import RetryConfig
from coolprompt.utils.enums import Task


class _SpecModel:
    def invoke(self, request: str) -> dict[str, object]:
        return {
            "task": "generation",
            "description": "Write a short answer.",
            "input_format": "Question text.",
            "output_format": "Answer text.",
            "requirements": [],
            "labels": None,
            "language": "English",
        }


class _GenerationModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: str) -> str:
        self.calls += 1
        count = 2 if "exactly 2" in request else 1

        inputs = {
            1: [
                "What causes ocean tides?",
                "How does photosynthesis work?",
            ],
            2: [
                "Why do metals expand when heated?",
            ],
        }

        examples = [
            {
                "input": inputs[self.calls][index],
                "output": f"answer-{self.calls}-{index}",
            }
            for index in range(count)
        ]

        return json.dumps({"examples": examples})


class _DistributionModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "axes": [
                    {
                        "name": "difficulty",
                        "description": "Question difficulty.",
                        "strategy": "balanced",
                        "values": [
                            {"id": "easy", "description": "Easy question."},
                            {"id": "hard", "description": "Hard question."},
                        ],
                    }
                ]
            }
        )


class _FeedbackGenerationModel:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[str] = []

    def invoke(self, request: str) -> str:
        self.calls += 1
        self.requests.append(request)

        batches = (
            [
                {
                    "input": "Why does ice float on water?",
                    "output": "Ice is less dense than liquid water.",
                    "axis_tags": {"difficulty": "easy"},
                },
                {
                    "input": "Why does ice float on water?",
                    "output": "Its solid structure lowers its density.",
                    "axis_tags": {"difficulty": "hard"},
                },
            ],
            [
                {
                    "input": "How do eclipses occur?",
                    "output": "One celestial body blocks another.",
                    "axis_tags": {"difficulty": "unknown"},
                },
                {
                    "input": "How does gravitational lensing reveal dark matter?",
                    "output": "Mass bends light and exposes otherwise unseen matter.",
                    "axis_tags": {"difficulty": "hard"},
                },
            ],
        )
        return json.dumps({"examples": batches[self.calls - 1]})


def test_generator_builds_spec_and_respects_batch_sizes_without_api_calls() -> None:
    generation_model = _GenerationModel()
    generator = SyntheticDataGenerator(
        model=generation_model,
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(
            max_retries=0,
            min_wait_seconds=0,
            max_wait_seconds=0,
        ),
    )

    result = generator.generate(
        prompt="Answer each question.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        detect_dataset=False,
        num_samples=3,
        batch_size=2,
        use_task_distribution=False,
        feedback_controlled=False,
    )

    assert result.dataset == [
        "What causes ocean tides?",
        "How does photosynthesis work?",
        "Why do metals expand when heated?",
    ]
    assert result.target == [
        "answer-1-0",
        "answer-1-1",
        "answer-2-0",
    ]
    assert result.context.spec.description == "Write a short answer."
    assert generation_model.calls == 2


def test_default_feedback_generation_covers_axes_deduplicates_and_tops_up() -> None:
    generation_model = _FeedbackGenerationModel()
    distribution_model = _DistributionModel()
    generator = SyntheticDataGenerator(
        model=generation_model,
        task_spec_model=_SpecModel(),
        distribution_model=distribution_model,
        retry_config=RetryConfig(
            max_retries=0,
            min_wait_seconds=0,
            max_wait_seconds=0,
        ),
    )

    result = generator.generate(
        prompt="Answer each science question.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        detect_dataset=False,
        num_samples=2,
        batch_size=2,
    )

    assert result.dataset == [
        "Why does ice float on water?",
        "How does gravitational lensing reveal dark matter?",
    ]
    assert generation_model.calls == 2
    assert distribution_model.calls == 1
    assert all('"axis_tags"' in request for request in generation_model.requests)
    assert generator.last_distribution is not None
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "difficulty": {"easy": 1, "hard": 1}
    }


def test_feedback_trimming_keeps_state_aligned_with_final_examples() -> None:
    distribution = TaskDistribution(
        axes=tuple(
            TaskAxis(
                name=axis,
                description=axis,
                values=(
                    AxisValue(id="0", description="First category."),
                    AxisValue(id="1", description="Second category."),
                ),
            )
            for axis in ("a", "b")
        )
    )
    proposals = (
        [
            ("Why does ice float?", "0", "1"),
            ("How do redwoods grow tall?", "0", "1"),
            ("What bends light around a star?", "1", "0"),
        ],
        [
            ("How do coral reefs form?", "0", "0"),
            ("What causes an aurora?", "1", "0"),
        ],
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, request: str) -> str:
            batch = proposals[self.calls]
            self.calls += 1
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": question,
                            "output": "A concise scientific answer.",
                            "axis_tags": {"a": a, "b": b},
                        }
                        for question, a, b in batch
                    ]
                }
            )

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )

    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=3,
        batch_size=3,
    )

    assert model.calls == 2
    assert len(result.examples) == 3
    assert generator.last_generation_state is not None
    assert not coverage_gaps(distribution, generator.last_generation_state, 3)[0]

    expected_state = GenerationState()
    tags_by_input = {
        question: {"a": a, "b": b} for batch in proposals for question, a, b in batch
    }
    for example in result.examples:
        expected_state.record(tags_by_input[example.input])
    assert generator.last_generation_state.axis_counts == expected_state.axis_counts


def test_conflicting_duplicate_tags_are_checked_per_candidate() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Question difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy question."),
                    AxisValue(id="hard", description="Hard question."),
                ),
            ),
        ),
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, request: str) -> str:
            self.calls += 1
            if self.calls == 1:
                examples = [
                    {
                        "input": "Why does ice float on water?",
                        "output": "Ice is less dense than water.",
                        "axis_tags": {"difficulty": "invalid"},
                    },
                    {
                        "input": "Why does ice float on water?",
                        "output": "Ice is less dense than water.",
                        "axis_tags": {"difficulty": "easy"},
                    },
                    {
                        "input": "How does gravitational lensing reveal dark matter?",
                        "output": "Its gravity bends light from background galaxies.",
                        "axis_tags": {"difficulty": "hard"},
                    },
                ]
            else:
                raise AssertionError("A valid second candidate must not need top-up")
            return json.dumps({"examples": examples})

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )

    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
    )

    assert model.calls == 1
    assert result.dataset == [
        "Why does ice float on water?",
        "How does gravitational lensing reveal dark matter?",
    ]
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "difficulty": {"easy": 1, "hard": 1}
    }


def test_feedback_generation_accepts_empty_output_when_spec_allows_it() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="answerability",
                description="Whether the question has an answer.",
                values=(
                    AxisValue(id="answerable", description="Answer is present."),
                    AxisValue(id="unanswerable", description="Answer is absent."),
                ),
            ),
        ),
    )

    class Model:
        def invoke(self, request: str) -> str:
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": "Context: The sky is blue. Question: What color is the sky?",
                            "output": "blue",
                            "axis_tags": {"answerability": "answerable"},
                        },
                        {
                            "input": "Context: The sky is blue. Question: Who painted it?",
                            "output": "",
                            "axis_tags": {"answerability": "unanswerable"},
                        },
                    ]
                }
            )

    class SpecModel(_SpecModel):
        def invoke(self, request: str) -> dict[str, object]:
            return super().invoke(request) | {"allow_empty_output": True}

    generator = SyntheticDataGenerator(
        model=Model(),
        task_spec_model=SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )
    result = generator.generate(
        prompt="Answer only when the context supports an answer; otherwise return an empty string.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
    )

    assert result.context.spec.allow_empty_output is True
    assert result.target == ["blue", ""]
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "answerability": {"answerable": 1, "unanswerable": 1}
    }


@pytest.mark.parametrize("strict_coverage", (False, True))
def test_feedback_generation_reports_incomplete_balanced_coverage(
    strict_coverage: bool,
) -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Question difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy question."),
                    AxisValue(id="hard", description="Hard question."),
                ),
            ),
        ),
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, request: str) -> str:
            self.calls += 1
            questions = (
                ["What causes tides?", "Why does ice float?"]
                if self.calls == 1
                else ["How does rain form?"]
            )
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": question,
                            "output": "A short answer.",
                            "axis_tags": {"difficulty": "easy"},
                        }
                        for question in questions
                    ]
                }
            )

    generator = SyntheticDataGenerator(
        model=Model(),
        task_spec_model=_SpecModel(),
        max_topup_attempts=1,
        retry_config=RetryConfig(0, 0, 0),
    )

    def generate():
        return generator.generate(
            prompt="Answer science questions.",
            draft=TaskSpecDraft(task=Task.GENERATION),
            task_distribution=distribution,
            num_samples=2,
            batch_size=2,
            strict_coverage=strict_coverage,
        )

    if strict_coverage:
        with pytest.raises(RuntimeError, match="Coverage incomplete"):
            generate()
    else:
        assert len(generate().examples) == 2

    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {"difficulty": {"easy": 2}}


def test_feedback_targets_gaps_in_trimmed_dataset() -> None:
    distribution = TaskDistribution(
        axes=tuple(
            TaskAxis(
                name=axis,
                description=f"Variation axis {axis}.",
                values=(
                    AxisValue(id="0", description="First type."),
                    AxisValue(id="1", description="Second type."),
                ),
            )
            for axis in ("a", "b", "c")
        )
    )
    proposals = (
        [
            ("How do tides arise?", "0", "0", "0"),
            ("Why is basalt dark?", "1", "0", "0"),
        ],
        [
            ("What causes lunar eclipses?", "0", "1", "1"),
        ],
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[str] = []

        def invoke(self, request: str) -> str:
            self.requests.append(request)
            batch = proposals[self.calls]
            self.calls += 1
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": question,
                            "output": "A concise answer.",
                            "axis_tags": {"a": a, "b": b, "c": c},
                        }
                        for question, a, b, c in batch
                    ]
                }
            )

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )

    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
        strict_coverage=True,
    )

    assert model.calls == 2
    assert result.dataset == ["Why is basalt dark?", "What causes lunar eclipses?"]
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        axis: {"0": 1, "1": 1} for axis in ("a", "b", "c")
    }


def test_post_target_stops_after_three_stalled_gap_batches() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Question difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy question."),
                    AxisValue(id="hard", description="Hard question."),
                ),
            ),
        )
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, request: str) -> str:
            self.calls += 1

            if self.calls == 1:
                examples = [
                    {
                        "input": question,
                        "output": "Easy answer.",
                        "axis_tags": {"difficulty": "easy"},
                    }
                    for question in (
                        "Why does ice float on liquid water?",
                        "Which process allows green plants to make sugar from sunlight?",
                    )
                ]
            else:
                examples = [
                    {
                        "input": f"Invalid targeted question {self.calls}",
                        "output": "Invalid answer.",
                        "axis_tags": {"difficulty": "unknown"},
                    }
                ]

            return json.dumps({"examples": examples})

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        max_topup_attempts=1,
        retry_config=RetryConfig(0, 0, 0),
    )

    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
    )

    assert model.calls == 4
    assert len(result.examples) == 2
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {"difficulty": {"easy": 2}}


def test_feedback_generation_supports_classification_label_axis() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="label",
                description="Classification label.",
                values=(
                    AxisValue(id="label:0", description="negative"),
                    AxisValue(id="label:1", description="positive"),
                ),
            ),
        )
    )

    class Model:
        def invoke(self, request: str) -> str:
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": "The device failed immediately.",
                            "output": "negative",
                            "axis_tags": {"label": "label:1"},
                        },
                        {
                            "input": "The service was excellent.",
                            "output": "positive",
                            "axis_tags": {"label": "label:0"},
                        },
                    ]
                }
            )

    generator = SyntheticDataGenerator(
        model=Model(),
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )
    result = generator.generate(
        prompt="Classify sentiment.",
        draft=TaskSpecDraft(
            task=Task.CLASSIFICATION,
            description="Classify sentiment.",
            labels=("negative", "positive"),
        ),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
    )

    assert result.target == ["negative", "positive"]
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "label": {"label:0": 1, "label:1": 1}
    }


def test_distribution_aware_generation_without_feedback() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Question difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy question."),
                    AxisValue(id="hard", description="Hard question."),
                ),
            ),
        )
    )

    class Model:
        def __init__(self) -> None:
            self.request = ""

        def invoke(self, request: str) -> str:
            self.request = request
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": "Why is the sky blue?",
                            "output": "Because shorter blue wavelengths scatter more.",
                            "axis_tags": {"difficulty": "easy"},
                        },
                        {
                            "input": "How does renormalization work?",
                            "output": "It absorbs scale-dependent divergences into parameters.",
                            "axis_tags": {"difficulty": "hard"},
                        },
                    ]
                }
            )

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )
    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=2,
        batch_size=2,
        feedback_controlled=False,
    )

    assert len(result.examples) == 2
    assert '"axis_tags"' in model.request
    assert generator.last_distribution == distribution
    assert generator.last_generation_state is None


def test_target_proportions_are_respected_in_feedback_state() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="kind",
                description="Question kind.",
                strategy="target_proportions",
                values=(
                    AxisValue(id="factual", description="Factual.", target_ratio=0.75),
                    AxisValue(id="causal", description="Causal.", target_ratio=0.25),
                ),
            ),
        )
    )

    class Model:
        def invoke(self, request: str) -> str:
            examples = (
                ("Which gas is most abundant in Earth's atmosphere?", "factual"),
                ("What is the chemical symbol for gold?", "factual"),
                ("Where is the Mariana Trench located?", "factual"),
                (
                    "Why does increasing pressure raise a liquid's boiling point?",
                    "causal",
                ),
            )
            return json.dumps(
                {
                    "examples": [
                        {
                            "input": question,
                            "output": f"Answer {index}",
                            "axis_tags": {"kind": value},
                        }
                        for index, (question, value) in enumerate(examples)
                    ]
                }
            )

    generator = SyntheticDataGenerator(
        model=Model(),
        task_spec_model=_SpecModel(),
        retry_config=RetryConfig(0, 0, 0),
    )
    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=4,
        batch_size=4,
        strict_coverage=True,
    )

    assert len(result.examples) == 4
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "kind": {"factual": 3, "causal": 1}
    }


def test_post_target_resets_stalls_after_partial_gap_progress() -> None:
    distribution = TaskDistribution(
        axes=(
            TaskAxis(
                name="difficulty",
                description="Question difficulty.",
                values=(
                    AxisValue(id="easy", description="Easy question."),
                    AxisValue(id="hard", description="Hard question."),
                ),
            ),
        )
    )

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, request: str) -> str:
            self.calls += 1
            if self.calls == 1:
                examples = [
                    {
                        "input": question,
                        "output": "Easy answer.",
                        "axis_tags": {"difficulty": "easy"},
                    }
                    for question in (
                        "What is the freezing point of water?",
                        "Which planet is closest to the Sun?",
                        "Where do penguins naturally live?",
                        "When does a solar eclipse occur?",
                    )
                ]
            elif self.calls == 2:
                examples = [
                    {
                        "input": "How does quantum tunneling depend on barrier width?",
                        "output": "Its probability decreases exponentially with width.",
                        "axis_tags": {"difficulty": "hard"},
                    },
                    {
                        "input": "Invalid target example",
                        "output": "Invalid.",
                        "axis_tags": {"difficulty": "unknown"},
                    },
                ]
            else:
                examples = [
                    {
                        "input": f"Invalid stalled example {self.calls}",
                        "output": "Invalid.",
                        "axis_tags": {"difficulty": "unknown"},
                    }
                ]
            return json.dumps({"examples": examples})

    model = Model()
    generator = SyntheticDataGenerator(
        model=model,
        task_spec_model=_SpecModel(),
        max_topup_attempts=1,
        retry_config=RetryConfig(0, 0, 0),
    )
    result = generator.generate(
        prompt="Answer science questions.",
        draft=TaskSpecDraft(task=Task.GENERATION),
        task_distribution=distribution,
        num_samples=4,
        batch_size=4,
    )

    assert model.calls == 5
    assert len(result.examples) == 4
    assert generator.last_generation_state is not None
    assert generator.last_generation_state.axis_counts == {
        "difficulty": {"easy": 3, "hard": 1}
    }


@pytest.mark.parametrize(
    ("num_samples", "batch_size", "message"),
    (
        (0, 1, "num_samples"),
        (101, 1, "num_samples"),
        (1, 0, "batch_size"),
        (True, 1, "num_samples"),
    ),
)
def test_generator_rejects_invalid_sizes(
    num_samples: int,
    batch_size: int,
    message: str,
) -> None:
    generator = SyntheticDataGenerator(
        model=_GenerationModel(),
        task_spec_model=_SpecModel(),
    )

    with pytest.raises(ValueError, match=message):
        generator.generate(
            prompt="Answer questions.",
            num_samples=num_samples,
            batch_size=batch_size,
        )


def test_generator_rejects_incompatible_feature_flags() -> None:
    generator = SyntheticDataGenerator(
        model=_GenerationModel(),
        task_spec_model=_SpecModel(),
    )

    with pytest.raises(ValueError, match="feedback_controlled"):
        generator.generate(
            prompt="Answer questions.",
            use_task_distribution=False,
            feedback_controlled=True,
        )

    with pytest.raises(ValueError, match="strict_coverage"):
        generator.generate(
            prompt="Answer questions.",
            feedback_controlled=False,
            strict_coverage=True,
        )
