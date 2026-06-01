"""Unit tests for meta_prompt (MetaPromptOptimizer, HyPERLightMethod)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import BenchmarkContext
from coolprompt.optimizer.hyper.meta_prompt import (
    HyPERLightMethod,
    MetaPromptOptimizer,
    Optimizer,
    _build_full_meta_prompt_template,
)
from coolprompt.utils.prompt_templates.hyper_templates import (
    MetaPromptBuilder,
    Recommendation,
    SECTION_RECOMMENDATIONS,
)


def test_build_full_meta_prompt_template_contains_placeholders():
    builder = MetaPromptBuilder()
    full = _build_full_meta_prompt_template(builder)
    assert "{QUERY}" in full
    assert "{META_INFO_BLOCK}" in full
    assert "<user_query>" in full


def test_meta_prompt_optimizer_default_template():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    assert "{QUERY}" in opt.meta_prompt
    assert opt.builder is not None


def test_meta_prompt_optimizer_custom_meta_prompt_string():
    model = MagicMock(spec=BaseLanguageModel)
    custom = "BODY {QUERY} {META_INFO_BLOCK}"
    opt = MetaPromptOptimizer(model=model, meta_prompt=custom)
    assert opt.meta_prompt == custom


def test_meta_prompt_optimizer_get_section():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    recs = opt.get_section(SECTION_RECOMMENDATIONS)
    assert isinstance(recs, list)


def test_meta_prompt_optimizer_update_section_rebuilds_template():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    before = opt.meta_prompt
    opt.update_section(
        SECTION_RECOMMENDATIONS,
        [Recommendation(section="general", text="Be concise")],
    )
    assert opt.meta_prompt != before or "Be concise" in opt.meta_prompt


def test_meta_prompt_optimizer_set_meta_prompt():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    opt.set_meta_prompt("X {QUERY} {META_INFO_BLOCK}")
    assert opt.meta_prompt.startswith("X")


def test_meta_prompt_optimizer_format_meta_info_in_prompt():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    filled = opt._format_meta_prompt("hello", foo="bar", n=1)
    assert "hello" in filled
    assert "foo" in filled
    assert "bar" in filled


def test_meta_prompt_optimizer_format_no_meta():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    filled = opt._format_meta_prompt("only_query")
    assert "only_query" in filled


@patch("coolprompt.optimizer.hyper.meta_prompt.get_model_answer_extracted")
def test_meta_prompt_optimizer_optimize_single(mock_get):
    model = MagicMock(spec=BaseLanguageModel)
    mock_get.return_value = "<result_prompt>improved</result_prompt>"
    opt = MetaPromptOptimizer(model=model)
    out = opt.optimize("task prompt", meta_info={"k": "v"})
    assert out == "improved"
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] is model


@patch("coolprompt.optimizer.hyper.meta_prompt.get_model_answer_extracted")
def test_meta_prompt_optimizer_optimize_n_prompts_list(mock_get):
    model = MagicMock(spec=BaseLanguageModel)
    mock_get.return_value = [
        "<result_prompt>a</result_prompt>",
        "<result_prompt>b</result_prompt>",
    ]
    opt = MetaPromptOptimizer(model=model)
    outs = opt.optimize("q", n_prompts=2)
    assert outs == ["a", "b"]


def test_meta_prompt_optimizer_process_tagged_string():
    model = MagicMock(spec=BaseLanguageModel)
    opt = MetaPromptOptimizer(model=model)
    out = opt._process_model_output("<result_prompt>ok</result_prompt>")
    assert out == "ok"


@patch("coolprompt.optimizer.hyper.meta_prompt.MetaPromptOptimizer")
def test_hyper_light_method_optimize_delegates(mock_opt_class):
    mock_opt = MagicMock()
    mock_opt.optimize.return_value = "done"
    mock_opt_class.return_value = mock_opt
    model = MagicMock(spec=BaseLanguageModel)
    method = HyPERLightMethod()
    ctx = {"a": 1}
    result = method.optimize(
        model=model,
        initial_prompt="p",
        problem_description="pd",
        meta_prompt_context=ctx,
    )
    assert result == "done"
    mock_opt_class.assert_called_once_with(model=model)
    mock_opt.optimize.assert_called_once()
    call_kw = mock_opt.optimize.call_args.kwargs
    assert call_kw["meta_info"]["a"] == 1
    assert call_kw["meta_info"]["problem_description"] == "pd"


@patch("coolprompt.optimizer.hyper.meta_prompt.MetaPromptOptimizer")
def test_hyper_light_method_problem_description_from_arg(mock_opt_class):
    mock_opt = MagicMock()
    mock_opt.optimize.return_value = "x"
    mock_opt_class.return_value = mock_opt
    method = HyPERLightMethod()
    method.optimize(
        model=MagicMock(spec=BaseLanguageModel),
        initial_prompt="p",
        problem_description="only_pd",
        meta_prompt_context=None,
    )
    meta = mock_opt.optimize.call_args.kwargs["meta_info"]
    assert meta["problem_description"] == "only_pd"


@patch("coolprompt.optimizer.hyper.meta_prompt.MetaPromptOptimizer")
def test_hyper_light_meta_context_wins_problem_description(mock_opt_class):
    mock_opt = MagicMock()
    mock_opt.optimize.return_value = "x"
    mock_opt_class.return_value = mock_opt
    method = HyPERLightMethod()
    method.optimize(
        model=MagicMock(spec=BaseLanguageModel),
        initial_prompt="p",
        problem_description="from_arg",
        meta_prompt_context={"problem_description": "from_ctx"},
    )
    meta = mock_opt.optimize.call_args.kwargs["meta_info"]
    assert meta["problem_description"] == "from_ctx"


@patch("coolprompt.optimizer.hyper.meta_prompt.MetaPromptOptimizer")
def test_hyper_light_run_configured_benchmark(mock_opt_class):
    mock_opt = MagicMock()
    mock_opt.optimize.return_value = "bench_out"
    mock_opt_class.return_value = mock_opt
    model = MagicMock(spec=BaseLanguageModel)
    ev = MagicMock(spec=Evaluator)
    ctx = BenchmarkContext(
        model=model,
        config={
            "meta_info": {"problem_description": "yaml_pd", "extra": "y"},
            "problem_description": "top_level",
        },
        dataset_split=([], [], [], []),
        test_dataset=[],
        test_target=[],
        evaluator=ev,
    )
    out = HyPERLightMethod().run_configured_benchmark(ctx, "start")
    assert out == "bench_out"
    meta = mock_opt.optimize.call_args.kwargs["meta_info"]
    assert meta["problem_description"] == "yaml_pd"
    assert meta["extra"] == "y"


def test_hyper_light_is_data_driven_and_name():
    m = HyPERLightMethod()
    assert m.is_data_driven() is False
    assert m.name == "hyper_light"


def test_optimizer_is_abc():
    assert issubclass(MetaPromptOptimizer, Optimizer)


def test_optimizer_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Optimizer(MagicMock(spec=BaseLanguageModel))
