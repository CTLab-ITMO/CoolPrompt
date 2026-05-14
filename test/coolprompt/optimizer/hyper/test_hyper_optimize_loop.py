"""Heavy-mocked test for one HyPER outer iteration (raises ``hyper.py`` coverage)."""

from collections import deque
from unittest.mock import MagicMock, patch

from coolprompt.evaluator.evaluator import EvalResultDetailed, FailedExampleDetailed
from coolprompt.optimizer.hyper.hyper import HyPEROptimizer
from coolprompt.utils.prompt_templates.hyper_templates import Recommendation


def _detailed(score: float, fail_on_idx: int = 0):
    fe = FailedExampleDetailed(
        instance="q",
        assistant_answer="bad",
        ground_truth="g",
        batch_index=fail_on_idx,
    )
    return EvalResultDetailed(
        aggregate_score=score,
        score_per_task=[0.0, 1.0],
        raw_outputs=["out0", "out1"],
        failed_examples=[fe],
    )


@patch("coolprompt.optimizer.hyper.hyper.tqdm", lambda x, **kwargs: x)
@patch("coolprompt.optimizer.hyper.hyper.mmr_select")
@patch("coolprompt.optimizer.hyper.hyper._get_bertscore_evaluate")
@patch("coolprompt.optimizer.hyper.hyper.get_model_answer_extracted")
@patch(
    "coolprompt.optimizer.hyper.hyper.random.sample",
    lambda seq, k: seq[: min(k, len(seq))],
)
def test_hyper_optimizer_one_iteration_improves(
    mock_get_paraphrase, mock_bert, mock_mmr
):
    mock_get_paraphrase.return_value = ["paraphrased"]
    mock_bert.return_value = MagicMock()

    r0 = _detailed(0.7, 0)
    r1 = _detailed(0.5, 0)
    mock_mmr.return_value = [("start", r0), ("start\nparaphrased", r1)]

    ev = MagicMock()
    ev.metric = MagicMock()
    ev.metric.parse_output = lambda x: str(x)

    q: deque = deque([0.2, r0, r1, 0.99])

    def eval_side_effect(*_a, **kw):
        return q.popleft()

    ev.evaluate.side_effect = eval_side_effect

    model = MagicMock()
    opt = HyPEROptimizer(
        model=model,
        evaluator=ev,
        n_iterations=1,
        n_candidates=1,
        top_n_candidates=2,
        mini_batch_size=2,
        k_samples=1,
        random_seed=0,
        enable_instance_leak_audit=False,
    )

    with (
        patch.object(
            opt.feedback_module,
            "generate_recommendations",
            return_value=[Recommendation("general", "Do better")],
        ),
        patch.object(
            opt.feedback_module,
            "filter_recommendations",
            side_effect=lambda recs: recs,
        ),
        patch.object(opt.meta_prompt_module, "optimize", return_value="opt-final"),
    ):
        best, hist = opt.optimize(
            "start",
            (
                ["a", "b", "c", "d"],
                ["v1", "v2"],
                [0, 1, 0, 1],
                [0, 1],
            ),
            meta_info=None,
        )

    assert best == "opt-final"
    assert len(hist) == 1
    assert hist[0]["iteration"] == 1
