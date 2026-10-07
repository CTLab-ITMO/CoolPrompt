"""PromptTuner integration tests."""

import json
from types import SimpleNamespace

import pytest

from coolprompt.meta_selector import MetaSelectionResult
from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod
from coolprompt.utils.enums import Task


class _Method(AutoPromptingMethod):
    def __init__(self, method_name, data_driven=True):
        self._method_name = method_name
        self._data_driven = data_driven

    @property
    def name(self):
        return self._method_name

    def is_data_driven(self):
        return self._data_driven

    def optimize(self, **kwargs):
        return f"optimized with {self.name}"


class _Metric:
    def _get_name(self):
        return "bertscore"


class _Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, prompt, **kwargs):
        return 1.0 if "optimized" in prompt else 0.0


class _TaskDetector:
    def __init__(self, model):
        pass

    def generate(self, prompt):
        return Task.GENERATION.value


class _Rule:
    def __init__(self, model):
        pass


def _patch_runtime(monkeypatch, selection):
    import coolprompt.assistant as assistant
    import coolprompt.utils.var_validation as validation

    class _Selector:
        def __init__(self, *args, **kwargs):
            pass

        def select(self, **kwargs):
            return selection

        @classmethod
        def no_dataset_result(cls):
            return selection

    monkeypatch.setattr(assistant, "validate_model", lambda model: None)
    monkeypatch.setattr(assistant, "TaskDetector", _TaskDetector)
    monkeypatch.setattr(assistant, "Evaluator", _Evaluator)
    monkeypatch.setattr(assistant, "LanguageRule", _Rule)
    monkeypatch.setattr(assistant, "correct", lambda prompt, **kwargs: prompt)
    monkeypatch.setattr(
        assistant, "validate_and_create_metric", lambda *a, **k: _Metric()
    )
    monkeypatch.setattr(assistant, "APOMetaSelector", _Selector)
    monkeypatch.setattr(
        validation,
        "_METHOD_BY_NAME",
        {
            **validation._METHOD_BY_NAME,
            "rider": lambda: _Method("rider"),
            "hyper": lambda: _Method("hyper"),
            "hyper_light": lambda: _Method("hyper_light", data_driven=False),
        },
    )
    return assistant


def test_auto_uses_selected_rider_and_exposes_selection(monkeypatch):
    selection = MetaSelectionResult(
        profile={"domain": "news"},
        similar_datasets=["xsum"],
        candidates=[],
        recommended_method="RIDER",
        selected_method="rider",
        fallback_reason=None,
        recommended_model="openai/gpt-4o-mini",
        recommended_split="150/100/300",
        recommended_metric="BERTScore F1",
    )
    assistant = _patch_runtime(monkeypatch, selection)
    tuner = assistant.PromptTuner(target_model=object(), system_model=object())

    result = tuner.run(
        "Summarize: {text}",
        task="generation",
        dataset=["one", "two"],
        target=["a", "b"],
        method="auto",
        problem_description="Summarize news.",
        validation_size=0.5,
        enable_telemetry=False,
    )

    assert result == "optimized with rider"
    assert tuner.meta_selection is selection


def test_auto_applies_sapo_fallback_to_hyper(monkeypatch):
    selection = MetaSelectionResult(
        profile={"domain": "news"},
        similar_datasets=["xsum"],
        candidates=[],
        recommended_method="SAPO",
        selected_method="hyper",
        fallback_reason="SAPO is not implemented in CoolPrompt",
        recommended_model="openai/gpt-4o-mini",
        recommended_split="150/100/300",
        recommended_metric="BERTScore F1",
    )
    assistant = _patch_runtime(monkeypatch, selection)
    tuner = assistant.PromptTuner(target_model=object(), system_model=object())

    result = tuner.run(
        "Summarize: {text}",
        task="generation",
        dataset=["one", "two"],
        target=["a", "b"],
        method="auto",
        problem_description="Summarize news.",
        validation_size=0.5,
        enable_telemetry=False,
    )

    assert result == "optimized with hyper"
    assert (
        tuner.meta_selection.fallback_reason == "SAPO is not implemented in CoolPrompt"
    )


def test_auto_without_data_uses_hyper_light(monkeypatch):
    selection = MetaSelectionResult(
        profile={},
        similar_datasets=[],
        candidates=[],
        recommended_method="HyPER Light",
        selected_method="hyper_light",
        fallback_reason="No dataset and target were supplied; selected HyPER Light.",
        recommended_model=None,
        recommended_split=None,
        recommended_metric=None,
    )
    assistant = _patch_runtime(monkeypatch, selection)

    class _Generator:
        def __init__(self, model, **kwargs):
            pass

        def generate(self, **kwargs):
            return SimpleNamespace(
                dataset=["one", "two"],
                target=["a", "b"],
                context=SimpleNamespace(
                    spec=SimpleNamespace(description="Summarize news.")
                ),
            )

    monkeypatch.setattr(assistant, "SyntheticDataGenerator", _Generator)
    tuner = assistant.PromptTuner(target_model=object(), system_model=object())

    result = tuner.run(
        "Summarize: {text}",
        task="generation",
        method="auto",
        validation_size=0.5,
        enable_telemetry=False,
    )

    assert result == "optimized with hyper_light"
    assert tuner.meta_selection.selected_method == "hyper_light"


def test_corner_ratio_fails_before_reaching_optimizer(monkeypatch):
    selection = MetaSelectionResult(
        profile={},
        similar_datasets=[],
        candidates=[],
        recommended_method="HyPER Light",
        selected_method="hyper_light",
        fallback_reason=None,
        recommended_model=None,
        recommended_split=None,
        recommended_metric=None,
    )
    assistant = _patch_runtime(monkeypatch, selection)
    tuner = assistant.PromptTuner(target_model=object(), system_model=object())

    with pytest.raises(ValueError, match="corner_ratio is no longer supported"):
        tuner.run(
            "Summarize: {text}",
            task="generation",
            method="hyper_light",
            corner_ratio=0.4,
            enable_telemetry=False,
        )


class _DefaultGenerationModel:
    def __init__(self) -> None:
        self.generation_calls = 0

    def invoke(self, request: str) -> str:
        if "You are an expert NLP task analyst" in request:
            return json.dumps(
                {
                    "task": "generation",
                    "description": "Answer a science question concisely.",
                    "input_format": "A science question.",
                    "output_format": "A concise answer.",
                    "requirements": [],
                    "labels": None,
                    "language": "English",
                }
            )

        if "You define coverage axes" in request:
            return json.dumps(
                {
                    "axes": [
                        {
                            "name": "topic",
                            "description": "Scientific topic.",
                            "strategy": "balanced",
                            "values": [
                                {"id": "physics", "description": "Physics."},
                                {"id": "biology", "description": "Biology."},
                            ],
                        }
                    ]
                }
            )

        self.generation_calls += 1
        batches = (
            [
                {
                    "input": "Why do objects fall?",
                    "output": "Gravity accelerates them toward Earth.",
                    "axis_tags": {"topic": "physics"},
                },
                {
                    "input": "Why do objects fall?",
                    "output": "Earth's gravity pulls them downward.",
                    "axis_tags": {"topic": "biology"},
                },
            ],
            [
                {
                    "input": "How do plants convert light into stored energy?",
                    "output": "Photosynthesis converts light into chemical energy.",
                    "axis_tags": {"topic": "biology"},
                }
            ],
        )
        return json.dumps({"examples": batches[self.generation_calls - 1]})


def test_prompt_tuner_uses_default_feedback_controlled_generator(monkeypatch):
    selection = MetaSelectionResult(
        profile={},
        similar_datasets=[],
        candidates=[],
        recommended_method="HyPER Light",
        selected_method="hyper_light",
        fallback_reason=None,
        recommended_model=None,
        recommended_split=None,
        recommended_metric=None,
    )
    assistant = _patch_runtime(monkeypatch, selection)
    model = _DefaultGenerationModel()
    tuner = assistant.PromptTuner(target_model=model, system_model=model)

    result = tuner.run(
        "Answer each science question.",
        task="generation",
        method="hyper_light",
        generate_num_samples=2,
        validation_size=0.5,
        enable_telemetry=False,
    )

    assert result == "optimized with hyper_light"
    assert tuner.synthetic_dataset == [
        "Why do objects fall?",
        "How do plants convert light into stored energy?",
    ]
    assert model.generation_calls == 2
