from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    if size == "all":
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
    """Load datasets from ``config`` and build an evaluator"""

    data_split = config["dataset"]["configuration"].split("/")
    train_size = _parse_dataset_size(data_split[0])
    val_size = _parse_dataset_size(data_split[1])
    test_size = _parse_dataset_size(data_split[2])

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

    test_dataset, test_target = load_dataset(
        config["dataset"]["name"], size=test_size, split="test"
    )

    from coolprompt.utils.var_validation import validate_task

    task = validate_task(config["task"])
    metric = validate_and_create_metric(task, config["metric"])
    evaluator = Evaluator(
        model,
        task,
        metric,
        use_structured_output=config.get("use_structured_output", False)
    )

    return BenchmarkContext(
        model=model,
        config=config,
        dataset_split=dataset_split,
        test_dataset=test_dataset,
        test_target=test_target,
        evaluator=evaluator,
    )


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
        *,
        use_structured_output: bool = False,
        **kwargs,
    ) -> str:
        """Run the prompt optimization process.

        Args:
            model: Language model used by the optimizer.
            initial_prompt: Starting prompt to optimize.
            dataset_split: Optional ``(train, val, train_targets, val_targets)``
                split for data-driven methods.
            evaluator: Optional evaluator used to score prompts.
            problem_description: Optional natural-language task description.
            use_structured_output: If True, the underlying optimizer should use
                LangChain ``with_structured_output`` calls (JSON schema) for
                LLM interactions instead of free-form text parsing. Concrete
                methods that do not support structured output may raise
                ``NotImplementedError`` when this is True. Defaults to False.
            **kwargs: Method-specific extra parameters.
        """
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
        *,
        use_structured_output: bool = False,
    ) -> str:
        """Optimization step for YAML benchmarks; override where supported.

        Args:
            ctx: Pre-built :class:`BenchmarkContext` (datasets + evaluator).
            start_prompt: Prompt to start optimization from.
            use_structured_output: Centralized structured-output flag forwarded
                from :meth:`run`. Subclasses should propagate it to their
                underlying optimizer. Defaults to False.
        """
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
        use_structured_output = bool(
            config.get("method", {}).get("use_structured_output", False)
        )
        final_prompt = self.run_configured_benchmark(
            ctx,
            start_prompt,
            use_structured_output=use_structured_output,
        )
        val_score = ctx.evaluator.evaluate(
            prompt=final_prompt,
            dataset=ctx.dataset_split[1],
            targets=ctx.dataset_split[3],
        )
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
