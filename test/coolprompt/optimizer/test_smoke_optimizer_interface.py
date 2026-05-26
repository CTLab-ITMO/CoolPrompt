"""Smoke tests for optimizer method wiring after refactors."""

import pytest

from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod
from coolprompt.optimizer.hyper.meta_prompt import HyPERLightMethod
from coolprompt.optimizer.rider import RIDERGenesisMethod
from coolprompt.utils.var_validation import _METHOD_BY_NAME, validate_method


def test_autoprompting_module_exports():
    assert issubclass(HyPERLightMethod, AutoPromptingMethod)
    assert HyPERLightMethod().name == "hyper_light"
    assert issubclass(RIDERGenesisMethod, AutoPromptingMethod)
    assert RIDERGenesisMethod().name == "rider"
    assert RIDERGenesisMethod().is_data_driven() is False


def test_validate_method_string_class_and_instance_equivalent():
    a = validate_method("hyper_light")
    b = validate_method(HyPERLightMethod)
    c = validate_method(HyPERLightMethod())
    for m in (a, b, c):
        assert isinstance(m, HyPERLightMethod)
        assert m.name == "hyper_light"


def test_validate_rider_method():
    method = validate_method("rider")
    assert isinstance(method, RIDERGenesisMethod)
    assert method.name == "rider"


def test_validate_method_unknown_name():
    with pytest.raises(ValueError, match="Unknown method"):
        validate_method("not_a_real_method_key")


def test_validate_method_rejects_bad_type():
    with pytest.raises(TypeError, match="Method must be"):
        validate_method(42)


def test_validate_method_rejects_abstract_base_as_type():
    with pytest.raises(TypeError, match="concrete"):
        validate_method(AutoPromptingMethod)


def test_method_by_name_covers_expected_keys():
    assert set(_METHOD_BY_NAME.keys()) == {
        "hyper_light",
        "hyper",
        "reflective",
        "distill",
        "regps",
        "compress",
        "rider",
    }


def test_method_evaluation_entrypoint():
    from coolprompt.method_evaluation import method_evaluation as me

    assert callable(me.evaluate_method)
    assert hasattr(HyPERLightMethod(), "run")
    assert "rider" in me._BENCHMARK_IMPL


def test_prompt_tuner_importable():
    from coolprompt.assistant import PromptTuner

    assert PromptTuner.__name__ == "PromptTuner"
