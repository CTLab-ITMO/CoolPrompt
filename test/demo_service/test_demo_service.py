from __future__ import annotations

import time

import pytest

from demo_service.methods import METHODS, coerce_method_params
from demo_service.runner import run_comparison, run_single_optimization
from demo_service.schemas import CompareRequest, OptimizationRequest
from demo_service.settings import DemoSettings


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


def test_compare_runs_each_selected_method_with_own_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _FakeTuner.calls = []
    base = OptimizationRequest(
        start_prompt="Summarize text.",
        task="generation",
        method="hyper_light",
        metric="rouge",
        dataset=["a", "b", "c"],
        target=["a1", "b1", "c1"],
    )
    request = CompareRequest(
        base=base,
        methods=["hyper_light", "rider"],
        method_params_by_method={
            "hyper_light": {"use_structured_output": True},
            "rider": {"num_generations": 1, "population_size": 2},
        },
    )

    results = run_comparison(
        request,
        _settings(),
        tuner_factory=_fake_tuner_factory,
    )

    assert [item.method for item in results] == ["hyper_light", "rider"]
    assert _FakeTuner.calls[0]["method"] == "hyper_light"
    assert _FakeTuner.calls[0]["use_structured_output"] is True
    assert _FakeTuner.calls[1]["method"] == "rider"
    assert _FakeTuner.calls[1]["num_generations"] == 1
    assert _FakeTuner.calls[1]["population_size"] == 2


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


def test_app_job_api_mock_roundtrip(monkeypatch):
    monkeypatch.setenv("COOLPROMPT_DEMO_MOCK", "1")

    from fastapi.testclient import TestClient

    from demo_service import app as app_module

    app_module.settings = DemoSettings(force_mock=True)
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
