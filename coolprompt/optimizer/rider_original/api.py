"""One-call entry point: ``rider(model, prompt, inputs, targets)``."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import setup_logging
from coolprompt.utils.utils import get_stratified_dataset_split

from .method import RIDEROriginalMethod, build_config
from .optimizer import RIDEROriginalOptimizer, RIDERResult

_JUDGE_METRICS = ("llm_as_judge", "geval")


def rider(
    model: BaseLanguageModel,
    start_prompt: str,
    dataset: Sequence[str],
    target: Sequence[Any],
    *,
    task: str | Task = "classification",
    metric: str | None = None,
    system_model: BaseLanguageModel | None = None,
    problem_description: str | None = None,
    validation_size: float = 0.25,
    preset: str = "paper",
    verbose: bool = True,
    **options: Any,
) -> RIDERResult:
    """Optimize ``start_prompt`` with the original RIDER and return the full result.

    Args:
        model: model that solves the task (and writes prompts, unless
            ``system_model`` is given).
        start_prompt: the prompt to improve.
        dataset, target: labelled examples; split into train and validation.
        task: ``"classification"`` or ``"generation"``.
        metric: CoolPrompt metric name (``"f1"``, ``"em"``, ``"bertscore"``...);
            None uses the task default.
        system_model: model that runs the RIDER operators; defaults to ``model``.
        problem_description: what the task is; recommended, it guides the operators.
        validation_size: share of the data used to score prompts.
        preset: ``"paper"`` (N=12, G=8), ``"fast"`` or ``"smoke"``.
        verbose: print progress per generation.
        **options: any ``RIDERConfig`` field, e.g. ``num_generations=4``,
            ``validation_limit=50``, ``seed=7``.

    Returns:
        RIDERResult with ``best_prompt``, ``ensemble``, ``summary()``,
        ``save(path)`` and ``predict(inputs)`` (ensemble majority vote).
    """

    if verbose:
        setup_logging()
    task = Task(task)
    system_model = system_model or model
    config = build_config({"preset": preset, **options})
    metric_impl = validate_and_create_metric(
        task, metric, model=system_model if metric in _JUDGE_METRICS else None
    )
    evaluator = Evaluator(model, task, metric_impl)
    template = RIDEROriginalMethod().get_template(task)
    train_x, val_x, train_y, val_y = get_stratified_dataset_split(
        list(dataset),
        list(target),
        validation_size=validation_size,
        task=task,
        random_state=config.seed if config.seed is not None else 42,
    )
    optimizer = RIDEROriginalOptimizer(
        system_model, evaluator, config, template=template
    )
    result = optimizer.optimize(
        start_prompt,
        train_inputs=list(train_x),
        train_targets=list(train_y),
        val_inputs=list(val_x),
        val_targets=list(val_y),
        problem_description=problem_description,
    )
    result.predictor = (evaluator, template)
    return result
