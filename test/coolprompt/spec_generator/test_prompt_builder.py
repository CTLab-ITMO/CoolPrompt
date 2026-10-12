"""Tests for synthetic-generation prompt contracts."""

from types import SimpleNamespace

import pytest

from coolprompt.optimizer.brave import BRAVEMethod
from coolprompt.spec_generator.distribution import AxisValue, TaskAxis, TaskDistribution
from coolprompt.spec_generator.models import Example, GenerationContext, TaskSpec
from coolprompt.spec_generator.prompt_builder import GenerationPromptBuilder
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.snippets_templates import (
    TAGGED_GENERATION_OUTPUT_CONTRACT,
)


def _context() -> GenerationContext:
    return GenerationContext(
        spec=TaskSpec(
            task=Task.GENERATION,
            description="Answer the question.",
            input_format="Question text.",
            output_format="Answer text.",
        )
    )


def _classification_context() -> GenerationContext:
    return GenerationContext(
        spec=TaskSpec(
            task=Task.CLASSIFICATION,
            description="Classify sentiment.",
            input_format="Review text.",
            output_format="One lowercase label.",
            requirements=("Do not explain the answer.",),
            labels=("negative", "positive"),
            allow_empty_output=False,
        ),
        seed_examples=(Example(input="Great product", output="positive"),),
    )


def _distribution() -> TaskDistribution:
    return TaskDistribution(
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


def test_regular_prompt_keeps_untagged_output_contract() -> None:
    prompt = GenerationPromptBuilder().regular(_context(), 2)

    assert prompt.count("Return only:") == 1
    assert '"axis_tags"' not in prompt


def test_classification_prompt_contains_labels_requirements_and_examples() -> None:
    prompt = GenerationPromptBuilder().regular(_classification_context(), 3)

    assert "Generate exactly 3" in prompt
    assert "- negative" in prompt
    assert "- positive" in prompt
    assert "- Do not explain the answer." in prompt
    assert "Empty output is not allowed." in prompt
    assert "Great product" in prompt


def test_distribution_prompt_uses_single_tagged_output_contract() -> None:
    prompt = GenerationPromptBuilder().distribution_aware(
        _context(),
        2,
        _distribution(),
    )

    assert prompt.count("Return only:") == 1
    assert '"axis_tags": {"axis_name": "value_id"}' in prompt
    assert prompt.rstrip().endswith(TAGGED_GENERATION_OUTPUT_CONTRACT.strip())


def test_targeted_prompt_uses_single_tagged_output_contract() -> None:
    prompt = GenerationPromptBuilder().targeted(
        _context(),
        1,
        _distribution(),
        targets=(
            {
                "count": 1,
                "constraints": [
                    {
                        "axis": "difficulty",
                        "value_id": "hard",
                        "description": "Hard question.",
                    }
                ],
            },
        ),
    )

    assert prompt.count("Return only:") == 1
    assert '"axis_tags": {"axis_name": "value_id"}' in prompt
    assert prompt.rstrip().endswith(TAGGED_GENERATION_OUTPUT_CONTRACT.strip())


def test_targeted_prompt_renders_targets_avoid_and_accepted_examples() -> None:
    accepted = tuple(
        Example(input=f"accepted-{index}", output=f"answer-{index}")
        for index in range(12)
    )
    prompt = GenerationPromptBuilder().targeted(
        _context(),
        2,
        _distribution(),
        targets=(
            {
                "count": 2,
                "constraints": [
                    {
                        "axis": "difficulty",
                        "value_id": "hard",
                        "description": "Hard question.",
                    }
                ],
            },
        ),
        avoid=(
            {
                "axis": "difficulty",
                "value_id": "easy",
                "description": "Easy question.",
            },
        ),
        accepted_examples=accepted,
    )

    assert "2 examples targeting: difficulty=hard" in prompt
    assert "avoid overusing difficulty=easy" in prompt
    assert "accepted-0" in prompt
    assert "accepted-11" in prompt
    assert prompt.count('"input": "accepted-') == 10


def test_prompt_builder_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        GenerationPromptBuilder().regular(_context(), 0)


def test_benchmark_problem_description_receives_all_classification_labels(
    monkeypatch,
):
    dataset_split = (
        ["input-1", "input-2", "input-3", "input-4", "input-5", "input-6"],
        ["val-input"],
        ["common", "common", "common", "common", "common", "rare"],
        ["common"],
    )

    captured = {}

    def fake_generate_problem_description(
        model,
        prompt,
        *,
        task,
        examples=None,
        labels=None,
        **kwargs,
    ):
        captured["task"] = task
        captured["examples"] = examples
        captured["labels"] = labels
        return "Classify the input as common or rare."

    monkeypatch.setattr(
        "coolprompt.optimizer.brave.run.sample",
        lambda population, count: [0, 1, 2, 3, 4],
    )

    monkeypatch.setattr(
        "coolprompt.optimizer.brave.run.generate_problem_description",
        fake_generate_problem_description,
    )

    monkeypatch.setattr(
        BRAVEMethod,
        "optimize",
        lambda self, **kwargs: "optimized prompt",
    )

    ctx = SimpleNamespace(
        config={},
        dataset_split=dataset_split,
        evaluator=SimpleNamespace(task=Task.CLASSIFICATION),
        _system_model=object(),
        model=object(),
    )

    result = BRAVEMethod().run_configured_benchmark(
        ctx,
        "Classify the input.",
    )

    assert result == "optimized prompt"

    assert {target for _, target in captured["examples"]} == {"common"}

    assert captured["labels"] == ["common", "rare"]
    assert captured["task"] == Task.CLASSIFICATION
