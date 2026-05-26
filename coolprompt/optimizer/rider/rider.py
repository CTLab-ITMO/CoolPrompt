"""CoolPrompt wrapper for vendored RIDER Genesis Ultra."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, override

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.rider import _llm_shim
from coolprompt.optimizer.rider._vendor import load_rider_genesis
from coolprompt.utils.logging_config import logger


class RIDEROptimizer:
    """Run the byte-identical vendored RiderGenesis through LangChain models."""

    _RIDER_MODEL_ALIAS = "coolprompt/langchain"
    _MODE = "ultra"
    _CONSTRUCTION_LOCK = threading.Lock()

    def __init__(
        self,
        model: BaseLanguageModel,
        *,
        planner_model: Optional[BaseLanguageModel] = None,
        judge_model: Optional[BaseLanguageModel] = None,
        critic_model: Optional[BaseLanguageModel] = None,
        mode: Optional[str] = None,
        verbose: bool = False,
        rider_model_alias: str = _RIDER_MODEL_ALIAS,
    ) -> None:
        self.model = model
        self.planner_model = planner_model or model
        self.judge_model = judge_model or self.planner_model
        self.critic_model = critic_model or self.planner_model
        self.mode = self._normalize_mode(mode)
        self.verbose = verbose
        self.rider_model_alias = rider_model_alias
        self._last_rider: Any = None

    @classmethod
    def _normalize_mode(cls, mode: Optional[str]) -> str:
        normalized = cls._MODE if mode is None else str(mode).strip().lower()
        if normalized != cls._MODE:
            raise ValueError(
                "CoolPrompt exposes only RIDER Ultra. "
                "Remove rider_mode/mode from the config or set it to 'ultra'."
            )
        return cls._MODE

    def _build_model_mapping(self, rider_genesis_cls: type) -> Dict[str, BaseLanguageModel]:
        mapping: Dict[str, BaseLanguageModel] = {self.rider_model_alias: self.model}
        role_models = rider_genesis_cls._MODE_ROLE_MODELS.get(  # noqa: SLF001 - vendored API
            self._MODE,
            rider_genesis_cls._MODE_ROLE_MODELS.get("standard", {}),  # noqa: SLF001
        )
        role_to_model = {
            "worker": self.model,
            "planner": self.planner_model,
            "judge": self.judge_model,
            "critic": self.critic_model,
        }
        for role, langchain_model in role_to_model.items():
            for model_name in role_models.get(role, []):
                mapping[model_name] = langchain_model
        fallback = getattr(rider_genesis_cls, "FALLBACK_MODEL", None)
        if fallback:
            mapping[fallback] = self.model
        return mapping

    def optimize(self, prompt: str) -> str:
        rider_genesis_cls = load_rider_genesis()
        with self._CONSTRUCTION_LOCK:
            _llm_shim.register_models(
                self.model,
                self._build_model_mapping(rider_genesis_cls),
            )
            rider = rider_genesis_cls(
                model=self.rider_model_alias,
                api_key="-",
                verbose=self.verbose,
            )
        self._last_rider = rider
        logger.info("Running RIDER Genesis Ultra")
        return rider.run(prompt, mode=self._MODE)

    @property
    def api_calls(self) -> int:
        if self._last_rider is None:
            return 0
        return int(getattr(self._last_rider, "api_calls", 0))

    @property
    def last_rider(self) -> Any:
        return self._last_rider


class RIDERGenesisMethod(AutoPromptingMethod):
    """AutoPromptingMethod wrapper for RIDER Genesis Ultra."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        _ = (dataset_split, evaluator, problem_description)
        optimizer = RIDEROptimizer(
            model=model,
            planner_model=kwargs.pop("planner_model", kwargs.pop("planning_model", None)),
            judge_model=kwargs.pop("judge_model", None),
            critic_model=kwargs.pop("critic_model", None),
            mode=kwargs.pop("rider_mode", kwargs.pop("mode", None)),
            verbose=kwargs.pop("rider_verbose", kwargs.pop("verbose", False)),
            rider_model_alias=kwargs.pop("rider_model_alias", RIDEROptimizer._RIDER_MODEL_ALIAS),
        )
        return optimizer.optimize(initial_prompt)

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        method_config = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            mode=method_config.get("rider_mode", method_config.get("mode")),
            rider_verbose=method_config.get("verbose", False),
        )

    def is_data_driven(self) -> bool:
        return False

    @property
    @override
    def name(self) -> str:
        return "rider"
