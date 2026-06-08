"""CoolPrompt wrapper for RIDER Genesis Ultra."""

from __future__ import annotations

import random
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, override

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.rider import _llm_shim
from coolprompt.optimizer.rider._core_loader import load_rider_genesis
from coolprompt.utils.logging_config import logger

DatasetSplit = Tuple[List[str], List[str], List[Any], List[Any]]


class RIDEROptimizer:
    """Run RiderGenesis Ultra through LangChain models."""

    _RIDER_MODEL_ALIAS = "coolprompt/langchain"
    _DUMMY_API_KEY = "-"
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
        """Initialize the RIDER Genesis Ultra wrapper."""

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
        """Normalize and validate the public RIDER mode."""

        normalized = cls._MODE if mode is None else str(mode).strip().lower()
        if normalized != cls._MODE:
            raise ValueError(
                "CoolPrompt exposes only RIDER Ultra. "
                "Remove rider_mode/mode from the config or set it to 'ultra'."
            )
        return cls._MODE

    def _build_model_mapping(
        self,
        rider_genesis_cls: type,
    ) -> Dict[str, BaseLanguageModel]:
        """Build the RIDER model-name to LangChain-model mapping."""

        mapping: Dict[str, BaseLanguageModel] = {
            self.rider_model_alias: self.model,
        }
        role_models = rider_genesis_cls._MODE_ROLE_MODELS.get(  # noqa: SLF001
            self._MODE,
            rider_genesis_cls._MODE_ROLE_MODELS.get(  # noqa: SLF001
                "standard",
                {},
            ),
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

    @staticmethod
    def _coerce_list(value: Optional[Iterable[Any]]) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return list(value)

    @classmethod
    def _unpack_dataset_split(
        cls,
        dataset_split: Optional[
            Tuple[Iterable[str], Iterable[str], Iterable[Any], Iterable[Any]]
        ],
    ) -> DatasetSplit:
        if dataset_split is None:
            return ([], [], [], [])
        train_dataset, val_dataset, train_targets, val_targets = dataset_split
        return (
            cls._coerce_list(train_dataset),
            cls._coerce_list(val_dataset),
            cls._coerce_list(train_targets),
            cls._coerce_list(val_targets),
        )

    @staticmethod
    def _sample_pairs(
        dataset: Sequence[Any],
        targets: Sequence[Any],
        *,
        limit: Optional[int],
        seed: Optional[int],
    ) -> List[Tuple[Any, Any]]:
        if not dataset or not targets:
            return []
        size = min(len(dataset), len(targets))
        indices = list(range(size))
        if limit is not None and limit >= 0 and size > limit:
            rng = random.Random(seed)
            indices = sorted(rng.sample(indices, limit))
        return [(dataset[i], targets[i]) for i in indices]

    @staticmethod
    def _coerce_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def optimize(
        self,
        prompt: str,
        *,
        dataset_split: Optional[
            Tuple[Iterable[str], Iterable[str], Iterable[Any], Iterable[Any]]
        ] = None,
        evaluator: Optional[Evaluator] = None,
        problem_description: Optional[str] = None,
        template: Optional[str] = None,
        train_sample_size: int = 4,
        validation_sample_size: Optional[int] = None,
        external_eval_weight: float = 0.7,
        seed: Optional[int] = None,
        num_samples: Optional[int] = None,
        population_size: Optional[int] = None,
        num_generations: Optional[int] = None,
        use_llm_judge: Optional[bool] = None,
        temperature: Optional[float] = None,
        phase_temperatures: Optional[Dict[str, float]] = None,
        num_strategies: Optional[int] = None,
    ) -> str:
        """Optimize a prompt with the RIDER Genesis Ultra pipeline.

        ``dataset_split`` follows the CoolPrompt convention:
        ``(train_dataset, val_dataset, train_targets, val_targets)``. RIDER uses
        train examples as task context for mutation prompts and validation
        examples as an external CoolPrompt reranker during the Ultra beam.
        """

        train_dataset, val_dataset, train_targets, val_targets = (
            self._unpack_dataset_split(dataset_split)
        )
        train_limit = self._coerce_optional_int(train_sample_size, 4)
        train_examples = self._sample_pairs(
            train_dataset,
            train_targets,
            limit=max(0, train_limit or 0),
            seed=seed,
        )

        rider_genesis_cls = load_rider_genesis()
        with self._CONSTRUCTION_LOCK:
            _llm_shim.register_models(
                self.model,
                self._build_model_mapping(rider_genesis_cls),
            )
            rider = rider_genesis_cls(
                model=self.rider_model_alias,
                api_key=self._DUMMY_API_KEY,
                verbose=self.verbose,
            )

        if hasattr(rider, "configure_coolprompt_context"):
            rider.configure_coolprompt_context(
                problem_description=problem_description,
                train_examples=train_examples,
            )
        if evaluator is not None and val_dataset and val_targets:
            if hasattr(rider, "configure_external_evaluator"):
                rider.configure_external_evaluator(
                    evaluator=evaluator,
                    val_dataset=val_dataset,
                    val_targets=val_targets,
                    template=template,
                    max_examples=validation_sample_size,
                    weight=external_eval_weight,
                    seed=seed,
                )
            logger.info(
                "RIDER Ultra external validation enabled: %s validation samples",
                min(len(val_dataset), len(val_targets)),
            )
        else:
            logger.info("RIDER Ultra external validation disabled: no evaluator/val split")

        if hasattr(rider, "configure_hyperparameters"):
            rider.configure_hyperparameters(
                num_samples=num_samples,
                population_size=population_size,
                num_generations=num_generations,
                use_llm_judge=use_llm_judge,
                temperature=temperature,
                phase_temperatures=phase_temperatures,
                num_strategies=num_strategies,
            )

        run_kwargs = {
            "num_samples": num_samples,
            "population_size": population_size,
            "num_generations": num_generations,
            "use_llm_judge": use_llm_judge,
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}

        self._last_rider = rider
        logger.info("Running RIDER Genesis Ultra")
        return rider.run(prompt, mode=self._MODE, **run_kwargs)

    @property
    def api_calls(self) -> int:
        """Return API calls reported by the last RIDER Genesis run."""

        if self._last_rider is None:
            return 0
        return int(getattr(self._last_rider, "api_calls", 0))

    @property
    def last_rider(self) -> Any:
        """Return the last constructed ``RiderGenesis`` instance."""

        return self._last_rider


class RIDERGenesisMethod(AutoPromptingMethod):
    """AutoPromptingMethod wrapper for RIDER Genesis Ultra."""

    @staticmethod
    def _pop_first(kwargs: Dict[str, Any], *names: str, default: Any = None) -> Any:
        for name in names:
            if name in kwargs:
                return kwargs.pop(name)
        return default

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        """Optimize a prompt through the CoolPrompt ``AutoPromptingMethod`` API."""

        optimizer = RIDEROptimizer(
            model=model,
            planner_model=self._pop_first(kwargs, "planner_model", "planning_model"),
            judge_model=self._pop_first(kwargs, "judge_model"),
            critic_model=self._pop_first(kwargs, "critic_model"),
            mode=self._pop_first(kwargs, "rider_mode", "mode"),
            verbose=self._pop_first(kwargs, "rider_verbose", "verbose", default=False),
            rider_model_alias=self._pop_first(
                kwargs,
                "rider_model_alias",
                default=RIDEROptimizer._RIDER_MODEL_ALIAS,
            ),
        )
        optimizer_kwargs = {
            "template": self._pop_first(kwargs, "template"),
            "train_sample_size": self._pop_first(
                kwargs,
                "rider_train_sample_size",
                "train_sample_size",
                default=4,
            ),
            "validation_sample_size": self._pop_first(
                kwargs,
                "rider_validation_sample_size",
                "validation_sample_size",
            ),
            "external_eval_weight": self._pop_first(
                kwargs,
                "rider_external_eval_weight",
                "external_eval_weight",
                default=0.7,
            ),
            "seed": self._pop_first(kwargs, "rider_seed", "seed"),
            "num_samples": self._pop_first(kwargs, "rider_num_samples", "num_samples"),
            "population_size": self._pop_first(
                kwargs,
                "rider_population_size",
                "population_size",
            ),
            "num_generations": self._pop_first(
                kwargs,
                "rider_num_generations",
                "num_generations",
                "epochs",
            ),
            "use_llm_judge": self._pop_first(
                kwargs,
                "rider_use_llm_judge",
                "use_llm_judge",
            ),
            "temperature": self._pop_first(kwargs, "rider_temperature", "temperature"),
            "phase_temperatures": self._pop_first(
                kwargs,
                "rider_phase_temperatures",
                "phase_temperatures",
            ),
            "num_strategies": self._pop_first(
                kwargs,
                "rider_num_strategies",
                "num_strategies",
            ),
        }
        if kwargs:
            logger.debug("Unused RIDER kwargs after parsing: %s", sorted(kwargs))
        return optimizer.optimize(
            initial_prompt,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            **optimizer_kwargs,
        )

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run RIDER Genesis Ultra inside the benchmark harness."""

        method_config = ctx.config.get("method", {})
        if not isinstance(method_config, dict):
            method_config = {}
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=method_config.get(
                "problem_description",
                ctx.config.get("problem_description"),
            ),
            mode=method_config.get("rider_mode", method_config.get("mode")),
            rider_verbose=method_config.get("verbose", False),
            rider_num_samples=method_config.get("num_samples"),
            rider_population_size=method_config.get("population_size"),
            rider_num_generations=method_config.get("num_generations"),
            rider_train_sample_size=method_config.get("train_sample_size", 4),
            rider_validation_sample_size=method_config.get("validation_sample_size"),
            rider_external_eval_weight=method_config.get("external_eval_weight", 0.7),
            rider_seed=method_config.get("seed"),
            rider_temperature=method_config.get("temperature"),
            rider_phase_temperatures=method_config.get("phase_temperatures"),
            rider_num_strategies=method_config.get("num_strategies"),
        )

    def is_data_driven(self) -> bool:
        """Return whether RIDER consumes train/validation data when available."""

        return True

    @property
    @override
    def name(self) -> str:
        """Return the CoolPrompt registry key."""

        return "rider"
