"""Unit tests for hyper.py helpers and HyPEROptimizer wiring."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator.evaluator import EvalResultDetailed
from coolprompt.evaluator.metrics import BertScoreMetric
from coolprompt.optimizer.hyper import hyper as hyper_mod
from coolprompt.optimizer.hyper.hyper import (
    HyPERMethod,
    HyPEROptimizer,
    _adaptive_lambda,
    _compute_similarity_matrix,
    _get_bertscore_evaluate,
    mmr_select,
    sample_mini_batch,
    sample_mini_batch_with_indices,
)


def test_sample_mini_batch_respects_seed():
    ds = list("abcdef")
    tg = list(range(6))
    a, b = sample_mini_batch(ds, tg, 3, seed=42)
    c, d = sample_mini_batch(ds, tg, 3, seed=42)
    assert a == c and b == d
    assert len(a) == 3


def test_sample_mini_batch_caps_size():
    ds = ["a"]
    tg = [0]
    s, t = sample_mini_batch(ds, tg, 100)
    assert s == ["a"] and t == [0]


def test_sample_mini_batch_with_indices():
    ds = ["x", "y", "z"]
    tg = [1, 2, 3]
    s, t, idx = sample_mini_batch_with_indices(ds, tg, 2, seed=0)
    assert len(s) == len(t) == len(idx) == 2
    assert s[0] == ds[idx[0]]


def test_adaptive_lambda_in_range():
    assert 0.5 <= _adaptive_lambda(0.0) <= 0.9
    assert 0.5 <= _adaptive_lambda(1.0) <= 0.9
    assert _adaptive_lambda(0.0) > _adaptive_lambda(1.0)


def test_compute_similarity_matrix_single_prompt():
    mock_bs = MagicMock()
    m = _compute_similarity_matrix(["only"], mock_bs)
    assert m.shape == (1, 1)
    mock_bs.compute.assert_not_called()


def test_compute_similarity_matrix_pairs():
    mock_bs = MagicMock()
    mock_bs.compute.return_value = {"f1": [0.1, 0.2, 0.3]}
    prompts = ["a", "b", "c"]
    m = _compute_similarity_matrix(prompts, mock_bs)
    assert m.shape == (3, 3)
    assert m[0, 0] == 1.0
    np.testing.assert_allclose(m[0, 1], 0.1)
    np.testing.assert_allclose(m[1, 2], 0.3)


def test_mmr_select_returns_all_when_k_small():
    cands = ["a", "b"]
    results = [
        EvalResultDetailed(aggregate_score=0.1, failed_examples=[]),
        EvalResultDetailed(aggregate_score=0.2, failed_examples=[]),
    ]
    mock_bs = MagicMock()
    out = mmr_select(cands, results, top_n=5, lambda_=0.7, bertscore_evaluate=mock_bs)
    assert len(out) == 2
    mock_bs.compute.assert_not_called()


def test_mmr_select_orders_by_mmr():
    cands = ["p0", "p1", "p2"]
    results = [
        EvalResultDetailed(aggregate_score=0.9, failed_examples=[]),
        EvalResultDetailed(aggregate_score=0.8, failed_examples=[]),
        EvalResultDetailed(aggregate_score=0.7, failed_examples=[]),
    ]
    mock_bs = MagicMock()
    mock_bs.compute.return_value = {"f1": [0.99, 0.1, 0.5]}
    out = mmr_select(cands, results, top_n=2, lambda_=0.5, bertscore_evaluate=mock_bs)
    assert len(out) == 2
    prompts = [o[0] for o in out]
    assert "p0" in prompts


def test_get_bertscore_uses_bert_metric_internal():
    inner = MagicMock()
    metric = MagicMock(spec=BertScoreMetric)
    metric._metric = inner
    assert _get_bertscore_evaluate(metric) is inner


def test_get_bertscore_loads_evaluate_once():
    fake_mod = MagicMock()
    fake_metric = MagicMock()
    fake_metric.__class__ = object
    with patch.dict("sys.modules", {"evaluate": fake_mod}):
        hyper_mod._bertscore_evaluate = None
        fake_mod.load.return_value = "loaded_metric"
        a = _get_bertscore_evaluate(fake_metric)
        b = _get_bertscore_evaluate(fake_metric)
        assert a == b == "loaded_metric"
        fake_mod.load.assert_called_once_with("bertscore")


def test_hyper_optimizer_feedback_matches_builder_sections():
    model = MagicMock(spec=BaseLanguageModel)
    evaluator = MagicMock()
    evaluator.metric = MagicMock()
    evaluator.metric.parse_output = lambda x: x
    opt = HyPEROptimizer(model=model, evaluator=evaluator)
    assert (
        opt.feedback_module.section_specs
        == opt.meta_prompt_module.builder.config.section_specs
    )


@patch.object(HyPEROptimizer, "_get_variants_from_best", return_value=[])
def test_hyper_optimizer_early_exit_no_candidates(_mock_variants):
    model = MagicMock(spec=BaseLanguageModel)
    ev = MagicMock()
    ev.evaluate.return_value = 0.5
    opt = HyPEROptimizer(model=model, evaluator=ev, n_iterations=1)
    out, hist = opt.optimize(
        "prompt",
        (["t1"], ["v1"], [0], [0]),
        meta_info=None,
    )
    assert out == "prompt"
    assert hist == []


@patch("coolprompt.optimizer.hyper.hyper.get_model_answer_extracted")
def test_hyper_optimizer_get_variants(mock_get):
    model = MagicMock(spec=BaseLanguageModel)
    ev = MagicMock()
    ev.metric = MagicMock()
    ev.metric.parse_output = lambda x: x
    opt = HyPEROptimizer(model=model, evaluator=ev)
    mock_get.return_value = ["alt1", "alt2"]
    variants = opt._get_variants_from_best("base", n_candidates=2)
    assert variants[0] == "base"
    assert variants[1:] == ["alt1", "alt2"]


@patch("coolprompt.optimizer.hyper.hyper.HyPEROptimizer")
def test_hyper_method_passes_hyper_meta_info(mock_cls):
    mock_inst = MagicMock()
    mock_inst.optimize.return_value = ("final", [])
    mock_cls.return_value = mock_inst
    model = MagicMock(spec=BaseLanguageModel)
    ev = MagicMock()
    split = (["a"], ["b"], [0], [1])
    meta = {"problem_description": "pd", "x": 1}
    HyPERMethod().optimize(
        model=model,
        initial_prompt="p",
        dataset_split=split,
        evaluator=ev,
        problem_description="ignored_if_in_meta",
        hyper_meta_info=meta,
    )
    kw = mock_inst.optimize.call_args.kwargs
    assert kw["meta_info"]["problem_description"] == "pd"
    assert kw["meta_info"]["x"] == 1


def test_hyper_method_name_and_data_driven():
    m = HyPERMethod()
    assert m.name == "hyper"
    assert m.is_data_driven() is True


def test_hyper_method_get_template():
    from coolprompt.utils.enums import Task

    t = HyPERMethod().get_template(Task.CLASSIFICATION)
    assert isinstance(t, str)
    assert len(t) > 0
