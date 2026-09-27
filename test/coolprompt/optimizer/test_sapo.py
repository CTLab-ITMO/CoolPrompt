"""Unit and integration-style tests for the CoolPrompt SAPO adapter."""

from types import SimpleNamespace

import pytest

from coolprompt.meta_selector import APOMetaSelector
from coolprompt.optimizer.sapo import SAPOMethod, SAPOOptimizer


class _JSONModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "Decompose the prompt" in prompt:
            return '{"role":"","context":"","tasks":"Summarize",' \
                '"output_format":""}'
        if "Analyze the prompt" in prompt:
            return '{"weak_segments":["tasks"],"strong_segments":[],' \
                '"recommendations":{"tasks":"Make the instruction clear"}}'
        if "Generate 2 diverse" in prompt:
            return '{"prompts":["Write a clear one-sentence summary.",' \
                '"Write a concise summary."]}'
        raise AssertionError(f"Unexpected prompt: {prompt}")


class _Evaluator:
    def __init__(self):
        self.prompts = []

    def evaluate(self, prompt, dataset, targets, return_detailed=False):
        assert return_detailed is True
        assert len(dataset) == len(targets)
        self.prompts.append(prompt)
        score = 0.9 if "clear" in prompt.lower() else 0.2
        return SimpleNamespace(
            aggregate_score=score,
            score_per_task=[score] * len(dataset),
            raw_outputs=["summary"] * len(dataset),
        )


def test_sapo_optimizes_with_coolprompt_model_and_evaluator():
    telemetry = []
    optimizer = SAPOOptimizer(
        model=_JSONModel(),
        evaluator=_Evaluator(),
        n_iterations=2,
        n_candidates=2,
        early_stopping_rounds=1,
        retry_delay_seconds=0,
        telemetry_callback=lambda *args: telemetry.append(args),
    )

    result = optimizer.optimize(
        "Summarize the article.",
        (["train"], ["validation"], ["train summary"], ["val summary"]),
    )

    assert result == "Write a clear one-sentence summary."
    assert optimizer.history[1]["improved"] is True
    assert optimizer.history[-1]["val_score"] == pytest.approx(0.9)
    assert telemetry[0] == (0, 0.2, "Summarize the article.")
    assert telemetry[-1][1:] == (0.9, result)


def test_sapo_method_adapter_and_empty_training_validation():
    method = SAPOMethod()
    with pytest.raises(ValueError, match="non-empty training"):
        method.optimize(
            model=_JSONModel(),
            initial_prompt="Summarize.",
            dataset_split=([], [], [], []),
            evaluator=_Evaluator(),
            problem_description=None,
            n_iterations=0,
        )


def test_meta_selector_maps_sapo_without_fallback():
    assert APOMetaSelector._map_method("SAPO") == ("sapo", None)


def test_sapo_rejects_invalid_search_configuration():
    with pytest.raises(ValueError, match="n_candidates"):
        SAPOOptimizer(model=_JSONModel(), evaluator=_Evaluator(), n_candidates=0)

