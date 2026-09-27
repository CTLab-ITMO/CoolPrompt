"""PromptTuner integration tests for ``method='auto'``."""

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
            "sapo": lambda: _Method("sapo"),
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


def test_auto_runs_selected_sapo(monkeypatch):
    selection = MetaSelectionResult(
        profile={"domain": "news"},
        similar_datasets=["xsum"],
        candidates=[],
        recommended_method="SAPO",
        selected_method="sapo",
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

    assert result == "optimized with sapo"
    assert tuner.meta_selection.fallback_reason is None


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
        def __init__(self, model):
            pass

        def generate(self, **kwargs):
            return ["one", "two"], ["a", "b"], "Summarize news."

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
