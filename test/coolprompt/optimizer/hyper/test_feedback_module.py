"""Unit tests for hyper.feedback_module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from coolprompt.evaluator.evaluator import FailedExampleDetailed
from coolprompt.optimizer.hyper.feedback_module import (
    ContrastiveCandidate,
    FeedbackModule,
)
from coolprompt.utils.prompt_templates.hyper_templates import (
    GENERAL_SECTION,
    PromptSectionSpec,
    Recommendation,
)


def test_contrastive_candidate_dataclass():
    c = ContrastiveCandidate("p", 1.0, "raw", parsed_answer="x")
    assert c.parsed_answer == "x"


def test_feedback_module_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword"):
        FeedbackModule(MagicMock(), typo=1)


def test_feedback_module_rejects_bad_probability():
    with pytest.raises(ValueError, match="contrastive_probability"):
        FeedbackModule(MagicMock(), contrastive_probability=1.5)


def test_feedback_module_rejects_negative_char_budget():
    with pytest.raises(ValueError):
        FeedbackModule(MagicMock(), contrastive_max_answer_chars=-1)


def test_build_section_descriptions_default():
    fb = FeedbackModule(MagicMock())
    text = fb._build_section_descriptions()
    assert GENERAL_SECTION in text


def test_build_section_descriptions_with_specs():
    specs = [PromptSectionSpec(name="Role", description="r")]
    fb = FeedbackModule(MagicMock(), section_specs=specs)
    assert "Role" in fb._build_section_descriptions()


def test_truncate_head_tail():
    fb = FeedbackModule(MagicMock())
    long = "a" * 100
    out = fb._truncate_head_tail(long, 4, 4)
    assert "truncated" in out
    assert out.startswith("aaaa")


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_generate_recommendation_parses_json(mock_get):
    mock_get.return_value = '{"section": "general", "text": "Do thing"}'
    fb = FeedbackModule(MagicMock())
    rec = fb.generate_recommendation(
        prompt="p",
        instance="i",
        model_answer="bad",
        ground_truth="g",
    )
    assert rec.section == GENERAL_SECTION
    assert rec.text == "Do thing"


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_generate_recommendations_parallel_contrastive(mock_get):
    mock_get.return_value = '{"section": "general", "text": "fix"}'
    fb = FeedbackModule(MagicMock(), contrastive_probability=0.0)
    fe = FailedExampleDetailed(
        instance="x",
        assistant_answer="a",
        ground_truth="g",
        batch_index=0,
    )
    recs = fb.generate_recommendations(
        "prompt",
        [fe],
        contrastive_candidates_per_failure=[[]],
    )
    assert len(recs) == 1


def test_generate_recommendations_length_mismatch():
    fb = FeedbackModule(MagicMock())
    fe = FailedExampleDetailed("i", "a", ground_truth="g")
    with pytest.raises(ValueError, match="same length"):
        fb.generate_recommendations(
            "p",
            [fe],
            contrastive_candidates_per_failure=[[], []],
        )


def test_filter_recommendations_empty():
    fb = FeedbackModule(MagicMock())
    assert fb.filter_recommendations([]) == []


@patch.object(FeedbackModule, "_filter_section", return_value=[Recommendation("general", "x", 1)])
def test_filter_routes_by_section(_mock_filter):
    fb = FeedbackModule(MagicMock())
    out = fb.filter_recommendations(
        [
            Recommendation("Role", "a"),
            Recommendation("Role", "b"),
        ]
    )
    assert len(out) >= 1


def test_drop_instance_leaks_no_problem_description():
    fb = FeedbackModule(MagicMock())
    recs = [Recommendation("general", "r")]
    assert fb.drop_instance_leaks(recs, "") == recs
    assert fb.last_audit_trace == []


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_drop_instance_leaks_parses_verdicts(mock_get):
    fb = FeedbackModule(MagicMock())
    recs = [Recommendation("general", "keep me", weight=1)]
    verdicts = {"verdicts": [{"verdict": "KEEP", "text": ""}]}
    mock_get.return_value = json.dumps(verdicts)
    out = fb.drop_instance_leaks(recs, "some task description")
    assert len(out) == 1
    assert fb.last_audit_trace[0]["verdict"] == "KEEP"


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_drop_instance_leaks_parse_failure_keeps(mock_get):
    fb = FeedbackModule(MagicMock())
    recs = [Recommendation("general", "z")]
    mock_get.return_value = "not json"
    out = fb.drop_instance_leaks(recs, "pd")
    assert out == recs
    assert fb.last_audit_trace[0]["verdict"] == "FALLBACK_KEEP"


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_partition_section_single(mock_get):
    fb = FeedbackModule(MagicMock())
    out = fb._filter_section("general", [Recommendation("general", "only", weight=1)])
    assert len(out) == 1
    mock_get.assert_not_called()


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_llm_partition_into_groups_fallback(mock_get):
    mock_get.return_value = "garbage"
    fb = FeedbackModule(MagicMock())
    groups = fb._llm_partition_into_groups(["a", "b"])
    assert groups == [[0], [1]]


@patch("coolprompt.optimizer.hyper.feedback_module.get_model_answer_extracted")
def test_llm_partition_into_groups_valid(mock_get):
    mock_get.return_value = "[[0, 1]]"
    fb = FeedbackModule(MagicMock())
    groups = fb._llm_partition_into_groups(["x", "y"])
    assert [0, 1] in groups or groups == [[0, 1]]


def test_parse_synthesized_filter_response_none():
    fb = FeedbackModule(MagicMock())
    assert fb._parse_synthesized_filter_response("no json") is None


def test_parse_synthesized_filter_response_ok():
    fb = FeedbackModule(MagicMock())
    raw = '{"synthesized": [{"text": " t ", "weight": 2}]}'
    out = fb._parse_synthesized_filter_response(raw)
    assert out == [("t", 2)]


def test_pick_best_contrastive():
    fb = FeedbackModule(MagicMock())
    cands = [
        ContrastiveCandidate("a", 0.5, "r"),
        ContrastiveCandidate("b", 0.9, "r"),
    ]
    best = fb._pick_best_contrastive(cands, failing_score=0.3)
    assert best is not None and best.prompt == "b"


def test_pick_best_contrastive_none_qualify():
    fb = FeedbackModule(MagicMock())
    cands = [ContrastiveCandidate("a", 0.1, "r")]
    assert fb._pick_best_contrastive(cands, failing_score=0.5) is None
