from pathlib import Path

import pytest

from coolprompt.optimizer.rider import rider as rider_module
from coolprompt.optimizer.rider.rider import RIDERGenesisMethod, RIDEROptimizer


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


def test_rider_genesis_core_is_byte_identical_to_source():
    source = Path(r"C:\projects\rider\rider\assistant.py")
    target = (
        Path(r"C:\projects\CoolPrompt")
        / "coolprompt"
        / "optimizer"
        / "rider"
        / "core"
        / "assistant.py"
    )
    if not source.exists():
        pytest.skip("Local RIDER source tree is not available.")

    assert target.read_bytes() == source.read_bytes()


def test_rider_core_contains_only_ultra_algorithm_source():
    core = (
        Path(r"C:\projects\CoolPrompt")
        / "coolprompt"
        / "optimizer"
        / "rider"
        / "core"
    )

    files = {
        path.relative_to(core).as_posix()
        for path in core.rglob("*")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    }

    assert files == {"__init__.py", "assistant.py"}


def test_rider_method_rejects_light_mode_override():
    with pytest.raises(ValueError, match="only RIDER Ultra"):
        RIDERGenesisMethod().optimize(
            model=object(),
            initial_prompt="Improve this prompt",
            rider_mode="light",
        )
