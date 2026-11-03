"""Offline tests for the original RIDER: no network, no real models."""

import json
import random
import re
import threading
import zlib
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from coolprompt.evaluator.evaluator import EvalResultDetailed, FailedExampleDetailed
from coolprompt.optimizer.rider_original import (
    RIDERConfig,
    RIDEROriginalMethod,
    RIDEROriginalOptimizer,
)
from coolprompt.optimizer.rider_original.bandit import WindowedThompsonSelector
from coolprompt.optimizer.rider_original.config import OPERATOR_CLASSES, OPERATORS
from coolprompt.optimizer.rider_original.optimizer import quota_allocation
from coolprompt.optimizer.rider_original.diversity import (
    PromptEmbedder,
    greedy_kdpp,
    mu_plus_lambda_select,
)
from coolprompt.optimizer.rider_original.memory import LongTermMemory, extract_pattern
from coolprompt.optimizer.rider_original.population import Candidate
from coolprompt.utils.enums import Task
from coolprompt.utils.var_validation import validate_method

KEYWORDS = ("carefully", "verify", "format", "edge", "concise", "explicit")


class FakeSystemModel(BaseChatModel):
    """Writes prompts that mention a random subset of fitness keywords."""

    seed: int = 0
    calls: int = 0
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-rider"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = messages[-1].content if messages else ""
        with _LOCK:
            self.calls += 1
            self.prompts.append(text)
            rng = random.Random(zlib.crc32(text.encode()) ^ self.calls ^ self.seed)
        words = rng.sample(KEYWORDS, rng.randint(1, len(KEYWORDS)))
        body = (
            "Solve the task step by step and " + " and ".join(words)
            + f" while answering the question number {rng.randint(0, 10**6)} in the required output."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Reasoning.\n<prompt>{body}</prompt>"))])


_LOCK = threading.Lock()


class F1Metric:
    """Stands in for a classification metric (only the class name matters)."""


class RougeMetric:
    """Stands in for an open-ended generation metric."""


class FakeEvaluator:
    """Fitness grows with keywords present but never reaches 1, so errors remain."""

    def __init__(self, task: Task = Task.CLASSIFICATION, metric=None) -> None:
        self.task = task
        self.metric = metric or F1Metric()
        self.calls = 0

    def evaluate(self, prompt, dataset, targets, template=None, failed_examples=None, *, return_detailed=False, **_):
        self.calls += 1
        hits = sum(1 for k in KEYWORDS if k in prompt)
        score = hits / (len(KEYWORDS) + 1)
        per_task = [1.0 if i < round(score * len(dataset)) else 0.0 for i in range(len(dataset))]
        failures = [
            FailedExampleDetailed(
                instance=dataset[i], assistant_answer="wrong", model_answer_parsed="wrong",
                metric_value=0.0, ground_truth=targets[i], batch_index=i,
            )
            for i, s in enumerate(per_task) if s < 1.0
        ][: failed_examples or 0]
        if not return_detailed:
            return score
        return EvalResultDetailed(score, per_task, failures, ["x"] * len(dataset))


def _split(n=10):
    xs = [f"question {i}" for i in range(n)]
    ys = [f"answer {i}" for i in range(n)]
    return xs, xs[:6], ys, ys[:6]


def _optimizer(task=Task.CLASSIFICATION, metric=None, **overrides):
    settings = dict(population_size=6, num_generations=3, max_workers=4, seed=7,
                    data_aware_seeds=2, zero_order_seeds=2)
    cfg = RIDERConfig(**{**settings, **overrides})
    model = FakeSystemModel()
    model.prompts = []
    evaluator = FakeEvaluator(task, metric)
    opt = RIDEROriginalOptimizer(model, evaluator, cfg, embedder=PromptEmbedder(None))
    return opt, model, evaluator


def test_full_loop_keeps_population_and_never_loses_the_best():
    opt, model, evaluator = _optimizer()
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y, problem_description="Answer questions.")
    assert len(result.history) == 3
    assert len(result.population) == 6
    bests = [r.best_fitness for r in result.history]
    assert bests == sorted(bests), "(mu+lambda) retention must never lose the best prompt"
    assert result.best_fitness == max(c.fitness for c in result.population)
    # 3N offspring per generation are requested.
    assert all(sum(r.operators.values()) == 18 for r in result.history)
    assert 1 <= len(result.ensemble) <= 5
    assert result.ensemble[0].fitness == result.best_fitness


def test_error_cases_reach_the_operators():
    opt, model, _ = _optimizer()
    train_x, val_x, train_y, val_y = _split()
    opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                 val_inputs=val_x, val_targets=val_y)
    assert any("still fails on these validation examples" in p and "Expected: answer" in p for p in model.prompts)


def test_open_ended_metrics_disable_reflective_operators():
    opt, _, _ = _optimizer(task=Task.GENERATION, metric=RougeMetric())
    assert "lamarckian" not in opt._enabled_operators()
    assert "reflection_crossover" not in opt._enabled_operators()
    # GSM8K-style generation with exact match keeps them: there is one correct answer.
    opt_em, _, _ = _optimizer(task=Task.GENERATION)
    assert "lamarckian" in opt_em._enabled_operators()


def test_first_generation_follows_class_budget_shares():
    ops = list(OPERATORS)
    chosen = quota_allocation(ops, 100)
    by_class = {name: sum(chosen.count(op) for op in members)
                for name, (_, members) in OPERATOR_CLASSES.items()}
    assert by_class == {"eda": 42, "generative": 31, "evolutionary": 20, "reflective": 7}
    assert len(quota_allocation(ops, 36)) == 36
    opt, _, _ = _optimizer()
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y)
    first = result.history[0].operators
    assert first.get("eda_mutation", 0) + first.get("eda_rank_index", 0) in (7, 8)  # 42% of 18


def test_reward_is_beating_the_parent_median():
    opt, _, _ = _optimizer(num_generations=1)
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y)
    stats = result.operator_statistics
    assert sum(s["trials"] for s in stats.values()) == 18
    assert any(s["mean_reward"] > 0.3 for s in stats.values())


def test_failed_evaluation_scores_minus_inf_and_run_continues():
    opt, _, evaluator = _optimizer(num_generations=1)
    original = evaluator.evaluate
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("provider timeout")
        return original(*args, **kwargs)

    evaluator.evaluate = flaky
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y)
    assert result.best_fitness > float("-inf")
    assert all(c.fitness != float("-inf") for c in result.population)


def test_stops_once_a_prompt_is_perfect():
    opt, _, _ = _optimizer(perfect_score=0.0)
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y)
    assert result.history == []


def test_prompt_extraction_and_degenerate_rejection():
    extract = RIDEROriginalOptimizer.extract_prompt
    assert extract("blah <prompt> Do X carefully. </prompt> tail") == ("Do X carefully.", True)
    assert extract("<think>hmm</think>Instruction: Do Y.") == ("Do Y.", False)
    assert extract("**New prompt:** Do Z.\n\nLet me know if it helps.") == ("Do Z.", False)
    opt, _, _ = _optimizer()
    assert opt._is_degenerate("Too short prompt.")
    assert opt._is_degenerate("Here is the improved prompt: " + "word " * 20)
    assert not opt._is_degenerate("Read the input and " + "answer precisely " * 8)


def test_thompson_selector_penalizes_and_forgets_old_failures():
    rng = random.Random(0)
    sel = WindowedThompsonSelector(["good", "bad"], {"good": 0.5, "bad": 0.5}, prior_strength=4,
                                   window=5, decay=0.9, penalty_threshold=0.3,
                                   penalty_min_trials=10, rng=rng)
    for g in range(1, 4):
        for _ in range(6):
            sel.update("good", g, 1.0)
            sel.update("bad", g, 0.0)
    picks = [sel.select(4, epsilon=0.0) for _ in range(200)]
    assert picks.count("good") > 190
    a, b = sel.posterior("bad", 4)
    assert b >= 10.0 and a < 1.0
    # Outside the window the failures no longer count (penalty aside).
    fresh = WindowedThompsonSelector(["x"], {"x": 0.5}, prior_strength=4, window=2, decay=0.9,
                                     penalty_threshold=0.3, penalty_min_trials=100, rng=rng)
    fresh.update("x", 1, 0.0)
    assert fresh.posterior("x", 10) == (3.0, 3.0)
    uniform = [sel.select(4, epsilon=1.0) for _ in range(400)]
    assert 120 < uniform.count("bad") < 280


def _sim_from_groups(groups):
    def sim(texts):
        import numpy as np
        return np.array([[1.0 if groups[a] == groups[b] else 0.0 for b in texts] for a in texts])
    return sim


def test_retention_uses_diversity_only_as_a_tiebreaker():
    a1, a2 = Candidate("A one", 0.800), Candidate("A two", 0.799)
    b = Candidate("B", 0.700)
    a3 = Candidate("A three", 0.500)
    sim = _sim_from_groups({"A one": "A", "A two": "A", "A three": "A", "B": "B"})
    kept = mu_plus_lambda_select([a2, b, a1, a3], 3, sim, threshold=0.92, tolerance=0.01)
    # a2 is a near-duplicate of a1 within 1%: skipped; a3 is similar but clearly worse: kept.
    assert [c.text for c in kept] == ["A one", "B", "A three"]
    kept = mu_plus_lambda_select([a2, b, a1], 3, sim, threshold=0.92, tolerance=0.01)
    assert {c.text for c in kept} == {"A one", "B", "A two"}, "fills with near-duplicates when short"


def test_kdpp_starts_from_the_best_and_prefers_diversity():
    cands = [Candidate("A1", 0.9), Candidate("A2", 0.89), Candidate("B", 0.6), Candidate("C", 0.5)]
    sim = _sim_from_groups({"A1": "A", "A2": "A", "B": "B", "C": "C"})
    chosen = greedy_kdpp(cands, 3, sim)
    assert chosen[0].text == "A1" and {c.text for c in chosen} == {"A1", "B", "C"}


def test_memory_extracts_first_sentence_and_actions():
    pattern = extract_pattern("Read the passage carefully. Then verify and return only the label.")
    assert pattern.startswith("Read the passage carefully.") and "verify" in pattern
    memory = LongTermMemory(patterns=1)
    kept = [Candidate("Good. Verify it.", 0.9), Candidate("Fine.", 0.5)]
    dropped = Candidate("Discarded offspring.", 0.05)
    memory.update(kept, kept + [dropped])
    text = memory.render()
    assert "Worked well" in text and "avoid" in text
    assert "Discarded offspring" in text, "failures come from everything evaluated"


def test_method_is_registered_and_validates_options():
    method = validate_method("rider_original")
    assert isinstance(method, RIDEROriginalMethod)
    assert method.name == "rider_original" and method.is_data_driven()
    with pytest.raises(ValueError):
        RIDERConfig(population_size=1).validate()


def test_method_runs_through_coolprompt_interface():
    method = RIDEROriginalMethod()
    model = FakeSystemModel()
    model.prompts = []
    best = method.optimize(model, "Answer the question.", dataset_split=_split(),
                           evaluator=FakeEvaluator(), rider_population_size=4,
                           rider_num_generations=1, data_aware_seeds=1, zero_order_seeds=1,
                           embedding_model=None)
    assert best == method.last_result.best_prompt
    assert method.last_result.system_calls > 0


class FlatEvaluator(FakeEvaluator):
    """Every prompt scores the same, so the search stagnates from the start."""

    def evaluate(self, prompt, dataset, targets, template=None, failed_examples=None, *, return_detailed=False, **_):
        return super().evaluate("carefully", dataset, targets, template, failed_examples,
                                return_detailed=return_detailed)


def test_stagnation_raises_temperature_once_then_restarts():
    opt, model, _ = _optimizer(num_generations=4)
    opt.evaluator = FlatEvaluator()
    train_x, val_x, train_y, val_y = _split()
    result = opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                          val_inputs=val_x, val_targets=val_y)
    temps = [round(r.temperature, 2) for r in result.history]
    assert temps == [0.7, 0.91, 0.91, 0.91], "T x1.3 once on entering stagnation"
    assert [r.epsilon for r in result.history] == [0.10, 0.10, 0.25, 0.25]
    assert [r.restarted for r in result.history] == [False, False, False, True]
    assert result.history[-1].stagnation == 0
    assert any(c.operator == "restart_zero_order" for c in result.population)


def test_open_ended_flag_overrides_the_metric():
    opt, _, _ = _optimizer(metric=RougeMetric(), open_ended=False)
    assert "lamarckian" in opt._enabled_operators()
    opt, _, _ = _optimizer(open_ended=True)
    assert "reflection_crossover" not in opt._enabled_operators()


class FakeTargetModel(BaseChatModel):
    """Answers correctly only when the instruction asks to verify."""

    @property
    def _llm_type(self) -> str:
        return "fake-target"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = messages[-1].content
        truth = "neg" if "bad" in text.split("Input:")[-1] else "pos"
        answer = truth if "verify" in text else "pos"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"<ans>{answer}</ans>"))])


def test_presets_and_validation_limit():
    assert RIDERConfig.from_preset("fast").num_generations == 4
    assert RIDERConfig.from_preset("fast", num_generations=2).num_generations == 2
    with pytest.raises(ValueError):
        RIDERConfig.from_preset("huge")
    from coolprompt.optimizer.rider_original.method import build_config
    cfg = build_config({"rider_preset": "smoke", "seed": None, "embedding_model": None})
    assert cfg.population_size == 4 and cfg.seed is None and cfg.embedding_model is None
    opt, _, evaluator = _optimizer(validation_limit=3, num_generations=1)
    sizes = []
    original = evaluator.evaluate
    evaluator.evaluate = lambda prompt, dataset, *a, **k: sizes.append(len(dataset)) or original(prompt, dataset, *a, **k)
    train_x, val_x, train_y, val_y = _split()
    opt.optimize("Answer the question.", train_inputs=train_x, train_targets=train_y,
                 val_inputs=val_x, val_targets=val_y)
    assert set(sizes) == {3}


def test_one_call_rider_api(tmp_path):
    from coolprompt import rider

    texts = [f"{'bad' if i % 2 else 'good'} review number {i}" for i in range(24)]
    labels = ["neg" if i % 2 else "pos" for i in range(24)]
    system = FakeSystemModel()
    system.prompts = []
    result = rider(FakeTargetModel(), "Label the review.", texts, labels, task="classification",
                   metric="f1", system_model=system, preset="smoke", embedding_model=None,
                   problem_description="Label reviews as pos or neg.", verbose=False)
    assert "verify" in result.best_prompt and result.best_fitness == 1.0
    assert result.initial_fitness is not None and result.initial_fitness < 1.0
    assert "RIDER: validation" in result.summary()
    saved = json.loads(result.save(tmp_path / "rider.json").read_text(encoding="utf-8"))
    assert saved["best_prompt"] == result.best_prompt
    assert result.predict(["bad thing", "good thing"]) == ["neg", "pos"]
