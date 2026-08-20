from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Protocol

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.enums import Task
from coolprompt.utils.load_dataset import load_dataset
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.utils import get_dataset_split


def _parse_dataset_size(size: str) -> Optional[int]:
    """Parse a dataset-size token; ``-`` and ``all`` mean no size limit."""

    if size in {"all", "-"}:
        return None
    return int(size)


@dataclass
class BenchmarkContext:
    """Train/val/test split and evaluator built from a YAML-style ``config``."""

    model: BaseLanguageModel
    config: dict[str, Any]
    dataset_split: tuple[list[str], list[str], list[str], list[str]]
    test_dataset: list[str]
    test_target: list[Any]
    evaluator: Evaluator

    @property
    def _system_model(self) -> BaseLanguageModel:
        return self.model


def build_benchmark_context(
    model: BaseLanguageModel, config: dict[str, Any]
) -> BenchmarkContext:
    """Load enabled dataset splits from ``config`` and build an evaluator.

    The dataset configuration is ``train/validation/test``. A ``-`` disables
    that split, so, for example, ``-/-/100`` loads only 100 test examples.
    """

    data_split = config["dataset"]["configuration"].split("/")
    if len(data_split) != 3:
        raise ValueError(
            "Dataset configuration must be train/validation/test, for example "
            "'10/10/100' or '-/-/100'."
        )

    train_requested, val_requested, test_requested = data_split
    train_size = _parse_dataset_size(data_split[0])
    val_size = _parse_dataset_size(data_split[1])
    test_size = _parse_dataset_size(data_split[2])

    if train_requested == "-" and val_requested == "-":
        dataset_split = ([], [], [], [])
    elif train_requested == "-":
        val_dataset, val_target = load_dataset(
            config["dataset"]["name"],
            size=val_size,
            split="train",
        )
        dataset_split = ([], val_dataset, [], val_target)
    elif val_requested == "-":
        train_dataset, train_target = load_dataset(
            config["dataset"]["name"],
            size=train_size,
            split="train",
        )
        dataset_split = (train_dataset, [], train_target, [])
    else:
        train_dataset, train_target = load_dataset(
            config["dataset"]["name"],
            size=train_size + val_size,
            split="train",
        )

        dataset_split = get_dataset_split(
            dataset=train_dataset,
            target=train_target,
            validation_size=val_size / (train_size + val_size),
            train_as_test=config.get("train_as_test", False),
        )

    if test_requested == "-":
        test_dataset, test_target = [], []
    else:
        test_dataset, test_target = load_dataset(
            config["dataset"]["name"], size=test_size, split="test"
        )

    from coolprompt.utils.var_validation import validate_task

    task = validate_task(config["task"])
    metric = validate_and_create_metric(task, config["metric"])
    evaluator = Evaluator(model, task, metric)

    return BenchmarkContext(
        model=model,
        config=config,
        dataset_split=dataset_split,
        test_dataset=test_dataset,
        test_target=test_target,
        evaluator=evaluator,
    )


class TelemetryCallback(Protocol):
    """Protocol for reporting iteration-level telemetry."""

    def __call__(self, iteration: int, best_score: float, best_prompt: str) -> None:
        pass


class AutoPromptingMethod(ABC):
    """Unified interface for auto‑prompting methods.

    This abstract base class defines the contract that all prompt
    optimization methods must implement. Concrete subclasses provide
    specific optimization strategies (e.g., evolutionary search,
    distillation, reflection, compression, etc.).
    """

    @abstractmethod
    def optimize(
        self,
        model: BaseLanguageModel,
        initial_prompt: str,
        dataset_split: Tuple[List[str], List[str], List[str], List[str]] | None,
        evaluator: Evaluator | None,
        problem_description: str | None,
        telemetry_callback: Optional[TelemetryCallback] = None,
        **kwargs,
    ) -> str:
        """Run the prompt optimization process."""
        pass

    @abstractmethod
    def is_data_driven(self) -> bool:
        """Whether this method needs a dataset and evaluator."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short method id (e.g. ``hyper_light``, ``reflective``)."""
        pass

    def get_template(self, task: Task) -> str:
        """Return the default prompt-formatting template for a task."""
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Optimization step for YAML benchmarks; override where supported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support method_evaluation benchmarks"
        )

    def run(
        self,
        model: BaseLanguageModel,
        config: dict[str, Any],
        start_prompt: str,
        *,
        saving_model_answers: bool = False,
    ) -> dict[str, Any]:
        """Load ``config`` splits, run :meth:`run_configured_benchmark`, score val/test.

        Returns:
            dict with keys ``final_prompt``, ``val_score``, ``test_score``.
        """
        ctx = build_benchmark_context(model, config)
        if not ctx.dataset_split[0] and self.is_data_driven():
            raise ValueError(
                "This method requires train/validation data; use a configuration "
                "such as '10/10/100' instead of '-/-/100'."
            )
        final_prompt = self.run_configured_benchmark(ctx, start_prompt)
        val_score = None
        if ctx.dataset_split[1]:
            val_score = ctx.evaluator.evaluate(
                prompt=final_prompt,
                dataset=ctx.dataset_split[1],
                targets=ctx.dataset_split[3],
            )
        test_score = None
        if ctx.test_dataset:
            test_score = ctx.evaluator.evaluate(
                prompt=final_prompt,
                dataset=ctx.test_dataset,
                targets=ctx.test_target,
                save_model_answers=saving_model_answers,
                model_answers_output_path=config.get(
                    "model_answers_output_path", "./model_answers.yaml"
                ),
            )
        return {
            "final_prompt": final_prompt,
            "val_score": val_score,
            "test_score": test_score,
        }
