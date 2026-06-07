from pathlib import Path

import pytest

from coolprompt.optimizer.rider import rider as rider_module
from coolprompt.optimizer.rider.rider import RIDERGenesisMethod, RIDEROptimizer
from coolprompt.utils.prompt_templates import rider_templates


class _FakeRiderGenesis:
    _MODE_ROLE_MODELS = {
        "standard": {
            "worker": ["worker-model"],
            "planner": ["planner-model"],
            "judge": ["judge-model"],
            "critic": ["critic-model"],
        },
        "ultra": {
            "worker": ["worker-model"],
            "planner": ["planner-model"],
            "judge": ["judge-model"],
            "critic": ["critic-model"],
        },
    }
    FALLBACK_MODEL = "fallback-model"
    instances = []

    def __init__(self, model=None, api_key=None, verbose=True):
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self.run_calls = []
        self.api_calls = 0
        self.__class__.instances.append(self)

    def configure_coolprompt_context(self, **kwargs):
        problem_description = kwargs.get("problem_description")
        train_examples = kwargs.get("train_examples") or []
        self._coolprompt_context_block = str(problem_description or "")
        if train_examples:
            self._coolprompt_context_block += f" examples={train_examples}"

    def configure_external_evaluator(self, **kwargs):
        self._external_eval_context = kwargs

    def configure_hyperparameters(self, **kwargs):
        self._rider_hyperparams = {k: v for k, v in kwargs.items() if v is not None}

    def run(self, prompt, mode="standard", **kwargs):
        self.run_calls.append((prompt, mode, kwargs))
        self.api_calls = 7
        return f"optimized via {mode}: {prompt}"


def test_rider_method_uses_ultra_by_default(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(
        rider_module,
        "load_rider_genesis",
        lambda: _FakeRiderGenesis,
    )

    result = RIDERGenesisMethod().optimize(
        model=object(),
        initial_prompt="Improve this prompt",
    )

    assert result == "optimized via ultra: Improve this prompt"
    instance = _FakeRiderGenesis.instances[-1]
    assert instance.model == RIDEROptimizer._RIDER_MODEL_ALIAS
    assert instance.api_key == RIDEROptimizer._DUMMY_API_KEY
    assert instance.verbose is False
    assert instance.run_calls == [("Improve this prompt", "ultra", {})]


def test_rider_optimizer_runs_ultra_and_reports_calls(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(
        rider_module,
        "load_rider_genesis",
        lambda: _FakeRiderGenesis,
    )

    optimizer = RIDEROptimizer(model=object(), mode="ultra", verbose=True)
    result = optimizer.optimize("Start prompt")

    assert result == "optimized via ultra: Start prompt"
    assert optimizer.api_calls == 7
    assert optimizer.last_rider is _FakeRiderGenesis.instances[-1]


def test_rider_optimizer_rejects_non_ultra_mode():
    with pytest.raises(ValueError, match="only RIDER Ultra"):
        RIDEROptimizer(model=object(), mode="light")


def test_load_rider_genesis_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cls = rider_module.load_rider_genesis()

    assert cls.__name__ == "RiderGenesis"
    assert hasattr(cls, "run")


def test_rider_genesis_uses_extracted_prompt_templates():
    cls = rider_module.load_rider_genesis()

    assert cls._STRATEGY_PROMPTS is rider_templates.RIDER_STRATEGY_PROMPTS
    assert cls._COMPARE_PROMPT is rider_templates.RIDER_COMPARE_PROMPT
    assert cls._MERGE_PROMPT is rider_templates.RIDER_MERGE_PROMPT
    assert cls._SYNTHETIC_TEST_PROMPT is rider_templates.RIDER_SYNTHETIC_TEST_PROMPT
    assert "structural" in cls._STRATEGY_PROMPTS
    assert "{prompt}" in cls._STRATEGY_PROMPTS["structural"]


def test_rider_core_loader_points_to_core_assistant():
    assert rider_module.load_rider_genesis.__module__.endswith("rider._core_loader")


def test_rider_core_modules_stay_reviewable():
    core_dir = Path(rider_module.__file__).resolve().parent / "core"
    modules = [
        path
        for path in core_dir.glob("*.py")
        if path.name not in {"__init__.py"}
    ]

    assert modules
    assert {
        "assistant.py",
        "runtime.py",
        "contract.py",
        "memory.py",
        "pipeline_config.py",
        "prompt_ops.py",
        "preservation.py",
        "run_modes.py",
        "ultra.py",
        "synthetic_eval.py",
        "schemas.py",
    } <= {path.name for path in modules}
    assert max(len(path.read_text(encoding="utf-8").splitlines()) for path in modules) <= 500


def test_rider_method_rejects_light_mode_override():
    with pytest.raises(ValueError, match="only RIDER Ultra"):
        RIDERGenesisMethod().optimize(
            model=object(),
            initial_prompt="Improve this prompt",
            rider_mode="light",
        )



class _FakeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, prompt, dataset, targets, template=None, **kwargs):
        self.calls.append((prompt, list(dataset), list(targets), template, kwargs))
        return 0.9 if "optimized" in prompt else 0.2


def test_rider_method_passes_dataset_and_hyperparameters(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(
        rider_module,
        "load_rider_genesis",
        lambda: _FakeRiderGenesis,
    )

    evaluator = _FakeEvaluator()
    result = RIDERGenesisMethod().optimize(
        model=object(),
        initial_prompt="Improve this prompt",
        dataset_split=(["train in"], ["val in"], ["train out"], ["val out"]),
        evaluator=evaluator,
        problem_description="Return a concise answer.",
        num_samples=3,
        population_size=4,
        num_generations=2,
        temperature=0.4,
        seed=13,
    )

    assert result == "optimized via ultra: Improve this prompt"
    instance = _FakeRiderGenesis.instances[-1]
    assert instance.run_calls == [
        (
            "Improve this prompt",
            "ultra",
            {
                "num_samples": 3,
                "population_size": 4,
                "num_generations": 2,
            },
        )
    ]
    assert instance._coolprompt_context_block
    assert "Return a concise answer" in instance._coolprompt_context_block
    assert instance._external_eval_context["evaluator"] is evaluator
    assert instance._external_eval_context["val_dataset"] == ["val in"]
    assert instance._rider_hyperparams["temperature"] == 0.4


def test_rider_benchmark_context_passes_dataset_and_config(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(
        rider_module,
        "load_rider_genesis",
        lambda: _FakeRiderGenesis,
    )

    class _Ctx:
        model = object()
        dataset_split = (["train"], ["val"], ["train-target"], ["val-target"])
        evaluator = _FakeEvaluator()
        config = {
            "problem_description": "Classify the input.",
            "method": {
                "num_samples": 2,
                "num_generations": 3,
                "population_size": 4,
                "validation_sample_size": 1,
                "external_eval_weight": 0.8,
                "seed": 42,
            },
        }

    result = RIDERGenesisMethod().run_configured_benchmark(
        _Ctx(),
        "Initial prompt",
    )

    assert result == "optimized via ultra: Initial prompt"
    instance = _FakeRiderGenesis.instances[-1]
    assert "Classify the input" in instance._coolprompt_context_block
    assert instance._external_eval_context["max_examples"] == 1
    assert instance._external_eval_context["weight"] == 0.8
    assert instance._external_eval_context["seed"] == 42
    assert instance._rider_hyperparams["num_generations"] == 3


def test_rider_external_validation_reranks_candidates():
    rider_cls = rider_module.load_rider_genesis()
    rider = object.__new__(rider_cls)
    rider._contract = {"task_archetype": "generation"}
    rider._synthetic_rankings = []
    rider._external_rankings = []
    rider._rider_hyperparams = {}
    rider._external_eval_context = {
        "evaluator": _FakeEvaluator(),
        "val_dataset": ["val a", "val b"],
        "val_targets": ["target a", "target b"],
        "template": None,
        "max_examples": 1,
        "weight": 1.0,
        "seed": 123,
    }

    def fake_synthetic(candidate, tests):
        return (0.9 if "baseline" in candidate else 0.1), []

    rider._evaluate_candidate_on_tests = fake_synthetic
    ranked = rider._rank_by_synthetic_eval(
        [("baseline", "baseline prompt"), ("optimized", "optimized prompt")],
        ["synthetic test"],
    )

    assert ranked[0][0][0] == "optimized"
    assert rider.external_rankings[-1]["num_examples"] == 1
    assert rider.external_rankings[-1]["combined_scores"][0]["name"] == "optimized"


def test_rider_budget_hyperparameters_change_ultra_plan():
    rider_cls = rider_module.load_rider_genesis()
    rider = object.__new__(rider_cls)
    base_cfg = {
        "num_strategies": 5,
        "tournament_k_phase1": 3,
        "run_crystallization": True,
        "run_validation": True,
        "run_red_team_harden": True,
        "run_triple_merge": True,
    }
    rider._rider_hyperparams = {
        "population_size": 4,
        "num_generations": 2,
    }

    cfg = rider._apply_budget_overrides(base_cfg)

    assert cfg["num_strategies"] == 4
    assert cfg["tournament_k_phase1"] == 2
    assert cfg["run_crystallization"] is True
    assert cfg["run_validation"] is False
    assert cfg["run_red_team_harden"] is False
    assert cfg["run_triple_merge"] is False
