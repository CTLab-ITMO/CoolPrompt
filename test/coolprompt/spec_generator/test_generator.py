import json

from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft
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
        examples = [
            {
                "input": f"question-{self.calls}-{index}",
                "output": f"answer-{self.calls}-{index}",
            }
            for index in range(count)
        ]
        return json.dumps({"examples": examples})


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

    assert result.dataset == ["question-1-0", "question-1-1", "question-2-0"]
    assert result.target == ["answer-1-0", "answer-1-1", "answer-2-0"]
    assert result.context.spec.description == "Write a short answer."
    assert generation_model.calls == 2
