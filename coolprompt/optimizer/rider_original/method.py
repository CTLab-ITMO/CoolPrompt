"""CoolPrompt ``AutoPromptingMethod`` wrapper for the original RIDER."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, override

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
    TelemetryCallback,
)
from coolprompt.utils.logging_config import logger

from .config import RIDERConfig
from .optimizer import RIDEROriginalOptimizer, RIDERResult

_CONFIG_FIELDS = {f.name for f in fields(RIDERConfig)}


def build_config(options: dict[str, Any]) -> RIDERConfig:
    """Build a config from keyword options; ``rider_``-prefixed keys also work."""

    values: dict[str, Any] = {}
    for key, value in options.items():
        name = key[len("rider_") :] if key.startswith("rider_") else key
        if name == "epochs":
            name = "num_generations"
        if name in _CONFIG_FIELDS and value is not None:
            values[name] = value
    if "disabled_operators" in values:
        values["disabled_operators"] = frozenset(values["disabled_operators"])
    return RIDERConfig(**values)


class RIDEROriginalMethod(AutoPromptingMethod):
    """RIDER as published at FRUCT: evolutionary search with nine LLM operators,
    Windowed Thompson Sampling operator selection, evaluate-first (mu + lambda)
    retention and error-directed evolution."""

    def __init__(self) -> None:
        self.last_result: RIDERResult | None = None

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        telemetry_callback: TelemetryCallback | None = None,
        **kwargs,
    ) -> str:
        """Optimize ``initial_prompt`` and return the best prompt found."""

        if evaluator is None or dataset_split is None:
            raise ValueError("RIDER requires a dataset split and an evaluator")
        train_x, val_x, train_y, val_y = dataset_split
        template = kwargs.pop("template", None)
        config = build_config(kwargs)
        unused = sorted(k for k in kwargs if k.removeprefix("rider_") not in _CONFIG_FIELDS | {"epochs"})
        if unused:
            logger.debug("RIDER: ignoring unknown options %s", unused)
        optimizer = RIDEROriginalOptimizer(model, evaluator, config, template=template)
        self.last_result = optimizer.optimize(
            initial_prompt,
            train_inputs=list(train_x),
            train_targets=list(train_y),
            val_inputs=list(val_x),
            val_targets=list(val_y),
            problem_description=problem_description,
            telemetry_callback=telemetry_callback,
        )
        logger.info(
            "RIDER finished: best validation score %.4f, %d system-model calls, "
            "%d prompt evaluations",
            self.last_result.best_fitness,
            self.last_result.system_calls,
            self.last_result.evaluations,
        )
        return self.last_result.best_prompt

    def run_configured_benchmark(self, ctx: BenchmarkContext, start_prompt: str) -> str:
        """Run RIDER inside the ``method_evaluation`` benchmark harness."""

        method_config = ctx.config.get("method", {})
        if not isinstance(method_config, dict):
            method_config = {}
        options = {k: v for k, v in method_config.items() if k != "problem_description"}
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=method_config.get(
                "problem_description", ctx.config.get("problem_description")
            ),
            **options,
        )

    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "rider_original"
