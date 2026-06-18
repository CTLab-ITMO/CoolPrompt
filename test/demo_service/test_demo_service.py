from __future__ import annotations

import time

import pytest

from demo_service.methods import METHODS, coerce_method_params
from demo_service.runner import _clean_prompt, _is_incomplete_prompt_response, run_single_optimization
from demo_service.schemas import OptimizationRequest
from demo_service.settings import DemoSettings, OPENROUTER_BASE_URL


class _FakeTuner:
    calls = []

    def __init__(self):
        self.init_metric = None
        self.final_metric = None
        self.init_prompt = None
        self.final_prompt = None
        self.synthetic_dataset = None
        self.synthetic_target = None

    def run(self, **kwargs):
        self.__class__.calls.append(kwargs)
        self.init_metric = 0.25
        self.final_metric = 0.75
        self.init_prompt = kwargs["start_prompt"]
        self.final_prompt = f"final::{kwargs['method']}::{kwargs['start_prompt']}"
        return self.final_prompt


def _fake_tuner_factory(request, settings):
    return _FakeTuner()


def _settings() -> DemoSettings:
    return DemoSettings(allow_mock=False, force_mock=False)


def test_openrouter_key_defaults_to_openrouter_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-test")

    settings = DemoSettings()

    assert settings.has_openai_key is True
    assert settings.openai_api_key == "sk-or-v1-test"
    assert settings.openai_base_url == OPENROUTER_BASE_URL


def test_openrouter_api_key_is_accepted_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    settings = DemoSettings()

    assert settings.has_openai_key is True
    assert settings.openai_api_key == "sk-or-v1-test"
    assert settings.openai_base_url == OPENROUTER_BASE_URL


def test_rider_ui_params_reach_prompt_tuner_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _FakeTuner.calls = []
    request = OptimizationRequest(
        start_prompt="Classify support tickets.",
        task="classification",
        method="rider",
        metric="f1",
        dataset=["refund", "crash", "delivery"],
        target=["billing", "technical", "shipping"],
        validation_size=0.34,
        method_params={
            "num_samples": 2,
            "num_generations": 3,
            "population_size": 4,
            "num_strategies": 3,
            "train_sample_size": 2,
            "validation_sample_size": 1,
            "external_eval_weight": 0.9,
            "temperature": 0.2,
            "seed": 7,
            "unsupported": "ignored",
        },
    )

    result = run_single_optimization(
        request,
        _settings(),
        tuner_factory=_fake_tuner_factory,
    )

    call = _FakeTuner.calls[-1]
    assert call["method"] == "rider"
    assert call["dataset"] == ["refund", "crash", "delivery"]
    assert call["target"] == ["billing", "technical", "shipping"]
    assert call["num_samples"] == 2
    assert call["num_generations"] == 3
    assert call["population_size"] == 4
    assert call["num_strategies"] == 3
    assert call["train_sample_size"] == 2
    assert call["validation_sample_size"] == 1
    assert call["external_eval_weight"] == 0.9
    assert call["temperature"] == 0.2
    assert call["seed"] == 7
    assert "unsupported" not in call
    assert result.final_metric == 0.75
    assert result.metric_delta == 0.5


def test_quality_guard_surfaces_initial_prompt_when_candidate_is_worse(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class WorseTuner:
        def __init__(self):
            self.init_metric = None
            self.final_metric = None
            self.init_prompt = None
            self.final_prompt = None
            self.synthetic_dataset = None
            self.synthetic_target = None

        def run(self, **kwargs):
            self.init_metric = 1.0
            self.final_metric = 0.5
            self.init_prompt = kwargs["start_prompt"]
            self.final_prompt = "worse candidate"
            return self.final_prompt

    request = OptimizationRequest(
        start_prompt="Classify support tickets.",
        task="classification",
        method="hyper_light",
        metric="f1",
        dataset=["refund", "crash"],
        target=["billing", "technical"],
    )

    result = run_single_optimization(
        request,
        _settings(),
        tuner_factory=lambda request, settings: WorseTuner(),
    )

    assert result.final_prompt == "Classify support tickets."
    assert result.final_metric == 1.0
    assert result.metric_delta == 0.0
    assert result.quality_guard


@pytest.mark.parametrize("method_meta", METHODS, ids=lambda item: item["id"])
def test_all_catalog_method_params_reach_prompt_tuner(monkeypatch, method_meta):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _FakeTuner.calls = []
    raw_params = {
        spec["name"]: (
            not bool(spec.get("default"))
            if spec.get("type") == "bool"
            else spec.get("default")
        )
        for spec in method_meta.get("params", [])
    }
    request = OptimizationRequest(
        start_prompt=f"Optimize with {method_meta['id']}",
        task="generation",
        method=method_meta["id"],
        metric="rouge",
        dataset=["input one", "input two", "input three"],
        target=["target one", "target two", "target three"],
        method_params={**raw_params, "unsupported": "ignored"},
    )

    run_single_optimization(
        request,
        _settings(),
        tuner_factory=_fake_tuner_factory,
    )

    call = _FakeTuner.calls[-1]
    assert call["method"] == method_meta["id"]
    for name, expected in coerce_method_params(method_meta["id"], raw_params).items():
        assert call[name] == expected
    assert "unsupported" not in call


def test_method_param_coercion_clamps_and_filters():
    params = coerce_method_params(
        "rider",
        {
            "num_generations": 100,
            "population_size": 1,
            "external_eval_weight": 2.5,
            "temperature": "0.35",
            "unknown": "drop",
        },
    )

    assert params["num_generations"] == 6
    assert params["population_size"] == 2
    assert params["external_eval_weight"] == 1.0
    assert params["temperature"] == 0.35
    assert "unknown" not in params


def test_clean_prompt_strips_outer_generation_wrappers():
    assert _clean_prompt("  <ans>Final prompt text</ans>  ") == "Final prompt text"
    assert _clean_prompt("[PROMPT_START]Final prompt text[PROMPT_END]") == "Final prompt text"
    assert _clean_prompt("<result_prompt>Final prompt text</result_prompt>") == "Final prompt text"
    assert _clean_prompt("<final_prompt>Final prompt text</final_prompt>") == "Final prompt text"
    assert _clean_prompt("Here is it:\n<result_prompt>Final prompt text</result_prompt>") == "Final prompt text"
    assert _clean_prompt("```markdown\nFinal prompt text\n```") == "Final prompt text"
    assert _clean_prompt("Use <ans>...</ans> in the answer") == "Use <ans>...</ans> in the answer"


def test_incomplete_prompt_response_is_detected():
    raw = "<result_prompt>\n# Role\nВы эксперт.\n\n# Task context\nВаша задача - создать резюме на"

    assert _is_incomplete_prompt_response(raw, _clean_prompt(raw)) is True
    assert _is_incomplete_prompt_response(
        "<result_prompt>Final prompt text</result_prompt>",
        _clean_prompt("<result_prompt>Final prompt text</result_prompt>"),
    ) is False


def test_quality_guard_surfaces_initial_prompt_when_model_returns_incomplete_wrapper(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class IncompleteTuner:
        def __init__(self):
            self.init_metric = None
            self.final_metric = None
            self.init_prompt = None
            self.final_prompt = None
            self.synthetic_dataset = None
            self.synthetic_target = None

        def run(self, **kwargs):
            self.init_metric = 0.8
            self.final_metric = 0.9
            self.init_prompt = kwargs["start_prompt"]
            self.final_prompt = (
                "<result_prompt>\n"
                "# Role\n"
                "Вы эксперт по деловой коммуникации.\n\n"
                "# Task context\n"
                "Ваша задача - создать краткое деловое резюме на"
            )
            return self.final_prompt

    request = OptimizationRequest(
        start_prompt="Сократи текст до короткого делового резюме.",
        task="generation",
        method="hyper_light",
        metric="llm_as_judge",
        dataset=["Длинное письмо клиента", "Отчет менеджера"],
        target=["Короткое резюме", "Краткий итог"],
    )

    result = run_single_optimization(
        request,
        _settings(),
        tuner_factory=lambda request, settings: IncompleteTuner(),
    )

    assert result.final_prompt == "Сократи текст до короткого делового резюме."
    assert result.final_metric == result.init_metric
    assert result.metric_delta == 0.0
    assert result.quality_guard


def test_mock_optimization_without_openai_key():
    request = OptimizationRequest(
        start_prompt="Improve this prompt",
        method="rider",
        task="generation",
        mock=True,
        dataset=["x", "y"],
        target=["a", "b"],
    )

    result = run_single_optimization(
        request,
        DemoSettings(allow_mock=True, force_mock=False),
    )

    assert result.used_mock is True
    assert result.method == "rider"
    assert result.dataset_size == 2
    assert "Optimized by RIDER" in result.final_prompt


def test_provider_error_is_sanitized_for_demo_ui():
    from demo_service.app import _public_error_message

    raw = (
        "Error code: 400 - {'error': {'message': 'The response was filtered due to "
        "the prompt triggering Azure OpenAI content management policy', "
        "'code': 'content_filter', 'metadata': {'raw': 'huge provider payload'}}}"
    )

    message = _public_error_message(RuntimeError(raw))

    assert "фильтром безопасности" in message
    assert "metadata" not in message
    assert "Azure OpenAI" not in message


def test_app_job_api_mock_roundtrip(monkeypatch):
    monkeypatch.setenv("COOLPROMPT_DEMO_ALLOW_MOCK", "1")

    from fastapi.testclient import TestClient

    from demo_service import app as app_module

    app_module.settings = DemoSettings(allow_mock=True, force_mock=False)
    client = TestClient(app_module.app)

    response = client.post(
        "/api/jobs",
        json={
            "mode": "single",
            "request": {
                "start_prompt": "Classify support tickets.",
                "task": "classification",
                "method": "rider",
                "metric": "f1",
                "dataset": ["refund", "crash"],
                "target": ["billing", "technical"],
                "method_params": {
                    "num_generations": 1,
                    "population_size": 2,
                },
                "mock": True,
            },
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    for _ in range(20):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.1)
    else:
        raise AssertionError("Mock job did not complete")

    assert status["result"]["method"] == "rider"
    assert status["result"]["method_params"]["num_generations"] == 1
    assert status["result"]["method_params"]["population_size"] == 2
    assert status["progress_stage"] == "completed"
    assert status["progress_percent"] == 100
    assert status["progress_message"] == "Оптимизация завершена"
