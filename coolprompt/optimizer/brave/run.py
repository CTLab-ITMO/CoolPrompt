"""High-level entry point for BRAVE prompt optimization."""

from dataclasses import asdict
from random import sample
from typing import Any, List, Mapping, Optional, Tuple, override

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.brave.evoluter import BRAVEEvoluter
from coolprompt.optimizer.brave.utils import BRAVEConfig
from coolprompt.spec_generator import SyntheticDataGenerator, TaskSpecDraft
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger


def brave(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: str,
    config: Optional[BRAVEConfig | Mapping[str, Any]] = None,
    seed: int = 19,
    verbose: bool = True,
    log_dir: Optional[str] = None,
    **config_overrides: Any,
) -> str:
    """Run BRAVE and return the best prompt on the validation split.

    ``config`` accepts either a :class:`BRAVEConfig` instance or a mapping.
    Additional keyword arguments override individual BRAVE configuration
    fields, matching the keyword-based API of the other CoolPrompt optimizers.
    """
    if config is None:
        config_values: dict[str, Any] = {}
    elif isinstance(config, BRAVEConfig):
        config_values = asdict(config)
    elif isinstance(config, Mapping):
        config_values = dict(config)
    else:
        raise TypeError("config must be a BRAVEConfig, a mapping, or None")

    config_values.update(config_overrides)
    brave_config = BRAVEConfig(**config_values)
    train_data, val_data, train_targets, val_targets = dataset_split

    evoluter = BRAVEEvoluter(
        model=model,
        evaluator=evaluator,
        config=brave_config,
        seed=seed,
        verbose=verbose,
        log_dir=log_dir,
    )
    logger.info("Starting BRAVE optimization...")
    result = evoluter.optimize(
        initial_prompt=initial_prompt,
        problem_description=problem_description,
        train_data=list(train_data),
        train_targets=list(train_targets),
        val_data=list(val_data),
        val_targets=list(val_targets),
    )
    logger.info("BRAVE optimization completed")
    return result["best_val_prompt"]


class BRAVEMethod(AutoPromptingMethod):
    """BRAVE implementation of the shared auto-prompting interface."""

    @override
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ) -> str:
        """Run BRAVE through the shared method interface."""
        kwargs.pop("telemetry_callback", None)
        return brave(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    @override
    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run BRAVE from a benchmark context."""
        problem_description = ctx.config.get("problem_description")
        if problem_description is None:
            generator = SyntheticDataGenerator(
                model=ctx._system_model,
                task_spec_model=ctx._system_model,
            )
            count = min(5, len(ctx.dataset_split[0]))
            indices = sample(range(len(ctx.dataset_split[0])), count)
            examples = [
                (ctx.dataset_split[0][index], str(ctx.dataset_split[2][index]))
                for index in indices
            ]
            labels = (
                tuple(dict.fromkeys(map(str, ctx.dataset_split[2])))
                if ctx.evaluator.task == Task.CLASSIFICATION
                else None
            )
            context = generator.build_context(
                prompt=start_prompt,
                draft=TaskSpecDraft(
                    task=ctx.evaluator.task,
                    labels=labels,
                ),
                examples=examples,
                detect_dataset=False,
            )
            problem_description = context.spec.description

        method_config = dict(ctx.config.get("method", {}))
        seed = method_config.pop("seed", 19)
        verbose = method_config.pop("verbose", True)
        log_dir = method_config.pop("log_dir", method_config.pop("output_path", None))
        return self.optimize(
            model=ctx.model,
            initial_prompt=start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=problem_description,
            config=method_config,
            seed=seed,
            verbose=verbose,
            log_dir=log_dir,
        )

    @override
    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "brave"
