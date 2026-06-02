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


def test_rider_method_rejects_light_mode_override():
    with pytest.raises(ValueError, match="only RIDER Ultra"):
        RIDERGenesisMethod().optimize(
            model=object(),
            initial_prompt="Improve this prompt",
            rider_mode="light",
        )
