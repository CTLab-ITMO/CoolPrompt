from pathlib import Path
from threading import Lock

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


class _ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.lock = Lock()

    def invoke(self, prompt, **kwargs):
        with self.lock:
            self.calls.append((prompt, dict(kwargs)))
            if not self.responses:
                raise RuntimeError("scripted model is out of responses")
            return self.responses.pop(0)


def test_rider_method_uses_ultra_by_default(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(rider_module, "load_rider_genesis", lambda: _FakeRiderGenesis)

    result = RIDERGenesisMethod().optimize(
        model=object(),
        initial_prompt="Improve this prompt",
    )

    assert result == "optimized via ultra: Improve this prompt"
    instance = _FakeRiderGenesis.instances[-1]
    assert instance.model == RIDEROptimizer._RIDER_MODEL_ALIAS
    assert instance.api_key == "-"
    assert instance.verbose is False
    assert instance.run_calls == [("Improve this prompt", "ultra", {})]


def test_rider_optimizer_accepts_mode_override_and_reports_calls(monkeypatch):
    _FakeRiderGenesis.instances = []
    monkeypatch.setattr(rider_module, "load_rider_genesis", lambda: _FakeRiderGenesis)

    optimizer = RIDEROptimizer(model=object(), mode="standard", verbose=True)
    result = optimizer.optimize("Start prompt")

    assert result == "optimized via standard: Start prompt"
    assert optimizer.api_calls == 7
    assert optimizer.last_rider is _FakeRiderGenesis.instances[-1]


def test_rider_optimizer_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown RIDER mode"):
        RIDEROptimizer(model=object(), mode="experimental")


def test_load_vendored_rider_genesis_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cls = rider_module.load_rider_genesis()

    assert cls.__name__ == "RiderGenesis"
    assert hasattr(cls, "run")


def test_vendored_rider_runtime_is_byte_identical_to_source():
    source = Path(r"C:\projects\rider\rider")
    target = Path(r"C:\projects\CoolPrompt\coolprompt\optimizer\rider\vendor\rider")
    if not source.exists():
        pytest.skip("Local RIDER source tree is not available.")

    source_files = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    target_files = {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }

    assert source_files.keys() == target_files.keys()
    changed = [rel for rel, content in source_files.items() if target_files[rel] != content]
    assert changed == []


def test_rider_light_end_to_end_with_real_vendored_flow():
    strategy_a = (
        "You are an expert prompt engineer. Write a four-line cat poem with concrete "
        "imagery, AABB rhyme, concise wording, and no commentary."
    )
    strategy_b = (
        "Act as a precise poetry coach. Produce exactly four lines about a cat, use "
        "AABB rhyme, sensory details, and output only the poem."
    )
    responses = [
        (
            '{"task_archetype":"creative_writing","language":"en","domain":"poetry",'
            '"audience":"general","output_format_anchor":"four-line poem",'
            '"must_preserve":[],"failure_modes":["generic wording"],'
            '"recommended_strategies":["structural","analytical"],"avoid_strategies":[]}'
        ),
        strategy_a,
        strategy_b,
        "WINNER: 2\nWHY: stronger constraints.",
        (
            "DIM1 CLARITY: 4\nDIM2 SPECIFICITY: 4\n"
            "DIM3 CONSTRAINT_COMPLETENESS: 4\nDIM4 OUTPUT_ANCHORING: 4\n"
            "DIM5 EDGE_CASE_COVERAGE: 3\nDIM6 BREVITY_VS_BLOAT: 4\n"
            "TOP_WEAKNESSES:\n"
            "- EDGE_CASE_COVERAGE: missing cliche guard -> add concrete anti-cliche guard"
        ),
        (
            "You are a precise poetry coach. Produce exactly four lines about a cat, "
            "use AABB rhyme, sensory details, avoid cliches, and output only the poem."
        ),
        "A: 3\nB: 9",
    ]
    model = _ScriptedModel(responses)

    optimizer = RIDEROptimizer(model=model, mode="light")
    output = optimizer.optimize("write a cat poem")

    assert "cat" in output.lower()
    assert "AABB" in output
    assert optimizer.api_calls == 7
    assert len(model.calls) == 7
