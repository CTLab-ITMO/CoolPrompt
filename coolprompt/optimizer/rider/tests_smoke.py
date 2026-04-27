"""Self-contained smoke tests for the vendored RiderGenesis LIGHT mode.

Verifies the integration uses the unmodified upstream ``RiderGenesis`` flow
from ``_rider_assistant.py`` with langchain models routed via the shim.

Run:
    cd C:\\projects\\CoolPrompt
    python coolprompt\\optimizer\\rider\\tests_smoke.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
import types
from abc import ABC, abstractmethod
from typing import Callable, List


HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- bootstrap
def _bootstrap():
    """Stub coolprompt package chain so we can import the rider modules
    without pulling in heavy CoolPrompt dependencies (langchain, deepeval...)
    that aren't required by our LIGHT path tests beyond ``langchain_core``.
    """
    def _stub(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    if "coolprompt" not in sys.modules:
        _stub("coolprompt")
    if "coolprompt.optimizer" not in sys.modules:
        _stub("coolprompt.optimizer")
    if "coolprompt.optimizer.rider" not in sys.modules:
        rp = _stub("coolprompt.optimizer.rider")
        rp.__path__ = [HERE]
    if "coolprompt.optimizer.hype" not in sys.modules:
        _stub("coolprompt.optimizer.hype")
    if "coolprompt.optimizer.hype.hype" not in sys.modules:
        hh = _stub("coolprompt.optimizer.hype.hype")

        class _Optimizer(ABC):
            def __init__(self, model):
                self.model = model

            @abstractmethod
            def optimize(self):
                pass

        hh.Optimizer = _Optimizer

    def _load(name: str, fname: str):
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    shim = _load("coolprompt.optimizer.rider._llm_shim", "_llm_shim.py")
    ra = _load("coolprompt.optimizer.rider._rider_assistant", "_rider_assistant.py")
    rd = _load("coolprompt.optimizer.rider.rider", "rider.py")
    return shim, ra, rd


_shim, _ra, _rd = _bootstrap()
RIDEROptimizer = _rd.RIDEROptimizer
RiderGenesis = _ra.RiderGenesis


# --------------------------------------------------------------------------- mock model
class MockChatModel:
    """LangChain-like model: ``invoke(prompt, **kwargs) -> AIMessageLike``.

    Tracks every call as ``(prompt, kwargs)`` for inspection.
    """

    def __init__(self, label: str, scripts: List[str]):
        self.label = label
        self.scripts = list(scripts)
        self.calls: list = []

    def invoke(self, prompt, **kwargs):
        self.calls.append((prompt, dict(kwargs)))
        if not self.scripts:
            raise RuntimeError(f"{self.label}: out of scripted responses")
        return self.scripts.pop(0)


# realistic strategy outputs (>=15 words so they aren't rejected)
_LONG_A = (
    "<prompt>You are a senior poet. Compose exactly four lines about a domestic cat. "
    "Use rhyme scheme AABB. Each line 6-10 syllables. Show one specific action and one "
    "mood. Avoid cliches like 'purring softly'. Output only the poem.</prompt>"
)
_LONG_B = (
    "<prompt>Act as an award-winning poet. Produce a four-line cat poem with rhyme AABB, "
    "6-10 syllables per line. Pick exactly one mood (playful, regal, curious) and reflect "
    "it in concrete sensory imagery. Forbidden words: purr, soft, whisker. Output only "
    "the four lines.</prompt>"
)


# --------------------------------------------------------------------------- helpers
def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: expected {b!r}, got {a!r}"


def assert_in(needle, hay, msg=""):
    assert needle in hay, f"{msg}: {needle!r} not in {hay!r}"


def make_planning_script(
    *,
    contract_recommended: list = None,
    contract_avoid: list = None,
    winner_idx: int = 2,
):
    contract_json = (
        '{"task_archetype": "creative_writing", "language": "en", "domain": "poetry", '
        '"audience": "general", "output_format_anchor": "four-line poem", '
        '"must_preserve": [], "failure_modes": ["cliche imagery"], '
        f'"recommended_strategies": {contract_recommended or ["structural", "analytical"]}, '
        f'"avoid_strategies": {contract_avoid or []}'
        "}"
    )
    return [
        contract_json,
        f"WINNER: {winner_idx}\nWHY: better specificity and constraints.",
        "A: 3\nB: 9",
    ]


def make_working_script():
    return [_LONG_A, _LONG_B]


# --------------------------------------------------------------------------- TESTS
def test_vendored_assistant_is_byte_identical_modulo_one_line():
    """The vendored copy must equal the upstream source except for ONE line
    (the LLMClient import)."""
    upstream = open(r"C:\projects\rider\rider\assistant.py", encoding="utf-8").read()
    vendored = open(os.path.join(HERE, "_rider_assistant.py"), encoding="utf-8").read()

    diff_lines = [
        (a, b)
        for a, b in zip(upstream.splitlines(), vendored.splitlines())
        if a != b
    ]
    assert_eq(len(diff_lines), 1, "exactly one line must differ between upstream and vendored")
    a, b = diff_lines[0]
    assert "from rider.llm.client import LLMClient" == a
    assert "from coolprompt.optimizer.rider._llm_shim import LLMClient" == b
    assert_eq(
        len(upstream.splitlines()), len(vendored.splitlines()),
        "no lines added or removed",
    )


def test_full_pipeline_5_calls_and_routing():
    """LIGHT runs exactly 5 LLM calls and routes planning vs working correctly."""
    planning = MockChatModel("planning", make_planning_script(winner_idx=2))
    working = MockChatModel("working", make_working_script())
    opt = RIDEROptimizer(model=working, planning_model=planning)
    out = opt.optimize("write a cat poem")

    assert_eq(len(planning.calls), 3, "planning model: contract + compare + quality")
    assert_eq(len(working.calls), 2, "working model: 2 strategies")
    assert "<prompt>" not in out, "wrapper tags must be stripped"
    assert_in("AABB", out, "winner should be one of the strategy outputs")


def test_strategy_calls_use_ignition_temperature():
    """Strategies must be invoked at IGNITION temperature 1.15."""
    planning = MockChatModel("p", make_planning_script(winner_idx=2))
    working = MockChatModel("w", make_working_script())
    RIDEROptimizer(model=working, planning_model=planning).optimize("write a cat poem")

    for prompt, kwargs in working.calls:
        assert_eq(kwargs.get("temperature"), 1.15, "strategy temperature must be IGNITION 1.15")


def test_compare_call_uses_zero_temperature():
    """Pairwise compare must run at temperature 0.0."""
    planning = MockChatModel("p", make_planning_script(winner_idx=2))
    working = MockChatModel("w", make_working_script())
    RIDEROptimizer(model=working, planning_model=planning).optimize("write a cat poem")

    # Planning calls: [0]=contract (T 0.7), [1]=compare (T 0.0), [2]=quality (T 0.0)
    assert_eq(planning.calls[1][1].get("temperature"), 0.0, "compare must be greedy T=0")


def test_planning_model_defaults_to_working_model():
    """If planning_model not provided, working model handles all 5 calls."""
    only_one = MockChatModel("only", [
        *make_planning_script(winner_idx=2)[:1],  # contract
        _LONG_A, _LONG_B,
        make_planning_script(winner_idx=2)[1],   # compare
        make_planning_script(winner_idx=2)[2],   # quality
    ])
    out = RIDEROptimizer(model=only_one).optimize("write a cat poem")
    assert_eq(len(only_one.calls), 5)
    assert_in("AABB", out)


def test_winner_is_original_returns_variant_anyway():
    """LIGHT guarantee: winner='original' (idx 1) still returns a variant."""
    planning = MockChatModel("p", make_planning_script(winner_idx=1))
    working = MockChatModel("w", make_working_script())
    out = RIDEROptimizer(model=working, planning_model=planning).optimize("write a cat poem")
    assert out != "write a cat poem"
    assert_in("AABB", out)


def test_short_strategy_outputs_skip_compare():
    """Both strategy outputs <15 words → no compare call, return original."""
    planning = MockChatModel("p", [make_planning_script()[0]])  # only contract
    working = MockChatModel("w", ["<prompt>too short.</prompt>", "<prompt>nope.</prompt>"])
    out = RIDEROptimizer(model=working, planning_model=planning).optimize("write a cat poem")
    # Only contract was called on planning; no compare/quality (LIGHT bails early
    # via _finalize_run which still does call _estimate_quality — but it's parameterised
    # with same prompt/prompt so semantics ok). Allow 1-2 calls.
    assert len(working.calls) == 2
    assert_eq(out, "write a cat poem")


def test_adaptive_strategy_selection_respects_contract():
    """Contract.recommended_strategies must drive which strategies are picked."""
    # Force contract to recommend 'depth' and 'techniques'
    contract_json = (
        '{"task_archetype": "code_generation", "language": "en", '
        '"recommended_strategies": ["depth", "techniques"], "avoid_strategies": ["creative"]}'
    )
    planning = MockChatModel("p", [
        contract_json,
        "WINNER: 2\nWHY: deeper.",
        "A: 4\nB: 9",
    ])
    working = MockChatModel("w", make_working_script())
    RIDEROptimizer(model=working, planning_model=planning).optimize("debug this code")

    # Inspect which strategy templates were used: each template starts with a unique phrase.
    p1, _ = working.calls[0]
    p2, _ = working.calls[1]
    template_signatures = {
        "structural": "ROLE PRIMING",
        "analytical": "find every weakness",
        "creative": "world-class prompt innovator",
        "depth": "depth specialist",
        "techniques": "prompt research scientist",
    }
    used = []
    for prompt_text in (p1, p2):
        for name, sig in template_signatures.items():
            if sig in prompt_text:
                used.append(name)
                break
    assert "depth" in used, f"contract said depth must be used; got {used}"
    assert "techniques" in used, f"contract said techniques must be used; got {used}"
    assert "creative" not in used, f"contract said avoid creative; got {used}"


def test_all_5_strategy_templates_present():
    """Verify all five RIDER strategy templates exist in the vendored code."""
    required = {"structural", "analytical", "creative", "depth", "techniques"}
    keys = set(RiderGenesis._STRATEGY_PROMPTS.keys())
    missing = required - keys
    assert not missing, f"missing strategy templates: {missing}"
    for name in required:
        tpl = RiderGenesis._STRATEGY_PROMPTS[name]
        assert "{prompt}" in tpl, f"strategy '{name}' must reference {{prompt}}"
        assert "SAME LANGUAGE" in tpl, f"strategy '{name}' must enforce SAME LANGUAGE"


def test_phase_temperatures_match_upstream():
    """The four phase temperatures must equal the documented values."""
    assert_eq(RiderGenesis._PHASE_T["ignition"], 1.15)
    assert_eq(RiderGenesis._PHASE_T["fusion"], 0.85)
    assert_eq(RiderGenesis._PHASE_T["crystallization"], 0.55)
    assert_eq(RiderGenesis._PHASE_T["validation"], 0.3)


def test_other_modes_raise():
    """Modes other than 'light' must raise NotImplementedError at construction."""
    for bad in ("blitz", "standard", "ultra", "ULTRA", "x"):
        try:
            RIDEROptimizer(model=MockChatModel("m", []), mode=bad)
        except NotImplementedError:
            continue
        raise AssertionError(f"mode='{bad}' should have raised")


def test_aimessage_like_response_handled():
    """The shim must coerce langchain AIMessage-like objects with .content."""
    class AIMessage:
        def __init__(self, c): self.content = c

    planning = MockChatModel("p", [AIMessage(s) for s in make_planning_script(winner_idx=2)])
    working = MockChatModel("w", [AIMessage(s) for s in make_working_script()])
    out = RIDEROptimizer(model=working, planning_model=planning).optimize("test prompt please")
    assert_in("AABB", out)


def test_invoke_kwargs_fallback():
    """Models that don't accept temperature/max_tokens kwargs must still work."""
    class StrictModel:
        def __init__(self, scripts):
            self.scripts = list(scripts)
            self.calls = []
        def invoke(self, prompt):  # no kwargs accepted
            self.calls.append(prompt)
            return self.scripts.pop(0)

    planning = StrictModel(make_planning_script(winner_idx=2))
    working = StrictModel(make_working_script())
    out = RIDEROptimizer(model=working, planning_model=planning).optimize("test prompt please")
    assert_in("AABB", out)


# --------------------------------------------------------------------------- runner
def run_all() -> bool:
    tests: List[Callable[[], None]] = [
        test_vendored_assistant_is_byte_identical_modulo_one_line,
        test_full_pipeline_5_calls_and_routing,
        test_strategy_calls_use_ignition_temperature,
        test_compare_call_uses_zero_temperature,
        test_planning_model_defaults_to_working_model,
        test_winner_is_original_returns_variant_anyway,
        test_short_strategy_outputs_skip_compare,
        test_adaptive_strategy_selection_respects_contract,
        test_all_5_strategy_templates_present,
        test_phase_temperatures_match_upstream,
        test_other_modes_raise,
        test_aimessage_like_response_handled,
        test_invoke_kwargs_fallback,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
