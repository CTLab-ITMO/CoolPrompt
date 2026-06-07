"""RIDER Genesis Ultra facade for CoolPrompt.

The heavy implementation is split into focused mixins under
``coolprompt.optimizer.rider.core`` so the public ``RiderGenesis`` class stays
readable while preserving the Ultra optimizer behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from coolprompt.utils.prompt_templates.rider_templates import (
    RIDER_BATCH_RANK_PROMPT,
    RIDER_BILINGUAL_ADVERSARIAL_PROMPT,
    RIDER_COMPARE_PROMPT,
    RIDER_CONSTITUTIONAL_AUDIT_PROMPT,
    RIDER_EVAL_RESPONSE_PROMPT,
    RIDER_MERGE_PROMPT,
    RIDER_PRESERVE_REPAIR_PROMPT,
    RIDER_QUALITY_PROMPT,
    RIDER_RED_TEAM_PROMPT,
    RIDER_REFINE_PROMPT,
    RIDER_STRATEGY_PROMPTS,
    RIDER_SYNTHETIC_TEST_PROMPT,
)
from rider.llm.client import LLMClient

from .contract import RiderContractMixin
from .memory import RiderMemoryMixin
from .pipeline_config import RiderPipelineConfigMixin
from .preservation import RiderPreservationMixin
from .prompt_ops import RiderPromptOpsMixin
from .run_modes import RiderRunModesMixin
from .runtime import RiderRuntimeMixin
from .synthetic_eval import RiderSyntheticEvalMixin
from .ultra import RiderUltraMixin

logger = logging.getLogger(__name__)


class RiderGenesis(
    RiderRuntimeMixin,
    RiderMemoryMixin,
    RiderContractMixin,
    RiderPipelineConfigMixin,
    RiderPromptOpsMixin,
    RiderPreservationMixin,
    RiderRunModesMixin,
    RiderUltraMixin,
    RiderSyntheticEvalMixin,
):
    """Elite automatic prompt optimization without labeled data."""

    # Backward-compatible constant for older imports. Real runs use the role
    # maps below so worker/planner/judge/critic can be different models.
    PLANNING_MODEL = "anthropic/claude-sonnet-4.6"

    _ROLES = ("worker", "planner", "judge", "critic")
    _MODE_ROLE_MODELS: Dict[str, Dict[str, List[str]]] = {
        "light": {
            "worker": ["anthropic/claude-sonnet-4.6", "google/gemini-3-flash-preview", "openai/gpt-5.4-mini"],
            "planner": ["openai/gpt-5.4-mini", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4.6"],
            "judge": ["google/gemini-3-flash-preview", "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4.6"],
            "critic": ["anthropic/claude-sonnet-4.6", "google/gemini-3-flash-preview", "openai/gpt-5.4-mini"],
        },
        "blitz": {
            "worker": ["anthropic/claude-sonnet-4.6", "google/gemini-3-flash-preview", "openai/gpt-5.4-mini"],
            "planner": ["openai/gpt-5.4-mini", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4.6"],
            "judge": ["google/gemini-3-flash-preview", "openai/gpt-5.4", "anthropic/claude-sonnet-4.6"],
            "critic": ["anthropic/claude-sonnet-4.6", "google/gemini-3-flash-preview", "openai/gpt-5.4-mini"],
        },
        "standard": {
            "worker": ["anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.7", "google/gemini-3-flash-preview"],
            "planner": ["google/gemini-3.1-pro-preview", "openai/gpt-5.5", "anthropic/claude-sonnet-4.6"],
            "judge": ["openai/gpt-5.5", "google/gemini-3.1-pro-preview", "anthropic/claude-sonnet-4.6"],
            "critic": ["anthropic/claude-opus-4.7", "google/gemini-3.1-pro-preview", "openai/gpt-5.4"],
        },
        "ultra": {
            "worker": ["anthropic/claude-opus-4.7", "anthropic/claude-sonnet-4.6"],
            "planner": ["google/gemini-3.1-pro-preview", "openai/gpt-5.5", "anthropic/claude-opus-4.7"],
            "judge": ["openai/gpt-5.5-pro", "openai/gpt-5.5", "google/gemini-3.1-pro-preview"],
            "critic": ["anthropic/claude-opus-4.7", "google/gemini-3.1-pro-preview", "openai/gpt-5.5", "x-ai/grok-4.3"],
        },
    }
    _BLOCKED_MODEL_PREFIXES = ("deepseek/",)
    _CONTENT_FILTER_FALLBACK_MODELS = (
        "qwen/qwen3.6-max-preview",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
    )

    # ULTRA-only: force max reasoning effort for top-tier Anthropic models.
    # Applied via OpenRouter ``extra_body={"reasoning":{"effort":"high"}}``.
    # Other modes keep default effort to avoid unnecessary thinking-token cost.
    _MAX_EFFORT_ULTRA_MODELS = frozenset({
        "anthropic/claude-opus-4.7",
    })

    # ══════════════════════════════════════════════════════════════════════
    # Strategy meta-prompts
    # ══════════════════════════════════════════════════════════════════════

    _STRATEGY_PROMPTS: Dict[str, str] = RIDER_STRATEGY_PROMPTS
    _COMPARE_PROMPT = RIDER_COMPARE_PROMPT
    _MERGE_PROMPT = RIDER_MERGE_PROMPT
    _CONSTITUTIONAL_AUDIT_PROMPT = RIDER_CONSTITUTIONAL_AUDIT_PROMPT
    _REFINE_PROMPT = RIDER_REFINE_PROMPT
    _PRESERVE_REPAIR_PROMPT = RIDER_PRESERVE_REPAIR_PROMPT
    _QUALITY_PROMPT = RIDER_QUALITY_PROMPT
    _BATCH_RANK_PROMPT = RIDER_BATCH_RANK_PROMPT

    # v4.3 role-chain fallback — used when primary model refuses (censor/safety content).
    FALLBACK_MODEL = "anthropic/claude-sonnet-4.6"
    _REFUSAL_PATTERNS = (
        r"I can'?t help with",
        r"I cannot (?:assist|help|provide|comply)",
        r"I'?m not able to (?:help|assist|provide)",
        r"I won'?t (?:help|assist|provide)",
        r"I must (?:decline|refuse)",
        r"unable to (?:assist|help|comply|provide|generate)",
        r"against my (?:guidelines|principles|policies)",
        r"cannot generate (?:this|such) content",
        r"Я не могу (?:помочь|выполнить|обработать)",
        r"не буду (?:помогать|обрабатывать|выполнять)",
    )

    _BILINGUAL_ADVERSARIAL_PROMPT = RIDER_BILINGUAL_ADVERSARIAL_PROMPT
    _SYNTHETIC_TEST_PROMPT = RIDER_SYNTHETIC_TEST_PROMPT
    _EVAL_RESPONSE_PROMPT = RIDER_EVAL_RESPONSE_PROMPT
    _RED_TEAM_PROMPT = RIDER_RED_TEAM_PROMPT

    # ══════════════════════════════════════════════════════════════════════
    # Construction
    # ══════════════════════════════════════════════════════════════════════

    # v4 Ultra+: persistent cross-prompt lesson cache (RIDER-like cross-experiment memory).
    _LESSON_CACHE_PATH = os.path.expanduser("~/.rider_genesis_lessons.json")
    _LESSON_CACHE_MAX_PER_KEY = 20

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize RIDER runtime clients, state, and persistent lesson cache."""
        self._model_override = model
        self._role_model_chains: Dict[str, List[str]] = {}
        self._instructor_clients: Dict[str, Any] = {}
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or self._MODE_ROLE_MODELS["standard"]["worker"][0]
        self.verbose = verbose

        if not self._api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY or pass api_key="
            )

        model_retries = max(1, int(os.environ.get("RIDER_GENESIS_MODEL_RETRIES", "2")))
        self.llm_client = LLMClient(
            provider="openrouter",
            api_key=self._api_key,
            max_retries=model_retries,
        )

        # Results (populated after run())
        self._final_prompt: Optional[str] = None
        self._original_fitness: float = 0.0
        self._best_fitness: float = 0.0
        self._history: List[Dict] = []
        self._api_calls_start: int = 0

        # v3 state — reset in _setup_run
        self._contract: Dict[str, Any] = {}
        self._lessons: List[str] = []
        self._forge: Dict[str, List[str]] = {}
        self._mode: str = "standard"
        self._original_prompt: str = ""
        self._synthetic_tests: List[str] = []
        self._synthetic_rankings: List[Dict[str, Any]] = []
        self._llm_attempts: List[Dict[str, Any]] = []

        # v4 cross-prompt persistent lesson cache.
        self._lesson_cache: Dict[str, List[str]] = self._load_lesson_cache()

    def run(
        self,
        prompt: str,
        mode: str = 'standard',
        num_samples: Optional[int] = None,
        population_size: Optional[int] = None,
        num_generations: Optional[int] = None,
        use_llm_judge: Optional[bool] = None,
    ) -> str:
        """Optimize an arbitrary prompt.

        Args:
            prompt: the original prompt to optimize
            mode: 'light' (~15s), 'blitz' (~45s), 'standard' (~70s), 'ultra' (~120s)
            num_samples: legacy kwarg (ignored in v3)
            population_size: legacy kwarg (ignored in v3)
            num_generations: legacy kwarg (ignored in v3)
            use_llm_judge: legacy kwarg (ignored in v3)

        Returns:
            The optimized prompt.
        """
        _ = (num_samples, population_size, num_generations, use_llm_judge)

        valid_modes = {'light', 'blitz', 'standard', 'ultra'}
        if mode not in valid_modes:
            logger.warning(
                f"RiderGenesis: unknown mode '{mode}', valid: {sorted(valid_modes)}. "
                f"Falling back to 'standard'."
            )
            mode = 'standard'

        if mode == 'light':
            return self.run_light(prompt)
        if mode == 'blitz':
            return self._run_blitz(prompt)
        if mode == 'ultra':
            return self._run_ultra(prompt)
        return self._run_standard(prompt)

    # -- Properties ---------------------------------------------------------

    @property
    def final_prompt(self) -> Optional[str]:
        return self._final_prompt

    @property
    def improvement(self) -> float:
        """Percentage fitness improvement from the last run."""
        if self._original_fitness <= 0:
            return 0.0
        return (self._best_fitness - self._original_fitness) / self._original_fitness * 100

    @property
    def fitness(self) -> float:
        return self._best_fitness

    @property
    def history(self) -> List[Dict]:
        return self._history

    @property
    def api_calls(self) -> int:
        return self.llm_client.total_api_calls - self._api_calls_start

    @property
    def contract(self) -> Dict[str, Any]:
        """Last extracted prompt contract (v3 addition)."""
        return dict(self._contract)
    @property
    def lessons(self) -> List[str]:
        """GENESIS-lite lessons collected during the last run (v3 addition)."""
        return list(self._lessons)
    @property
    def synthetic_tests(self) -> List[str]:
        """Synthetic cases generated during the last Standard/Ultra run."""
        return list(self._synthetic_tests)
    @property
    def synthetic_rankings(self) -> List[Dict[str, Any]]:
        """Synthetic beam audit trail: cases, candidate names, and scores."""
        return list(self._synthetic_rankings)
    @property
    def role_models(self) -> Dict[str, str]:
        """Primary model used by each RIDER Genesis role in the last/current run."""
        return {role: self._role_model(role) for role in self._ROLES}
    @property
    def llm_attempts(self) -> List[Dict[str, Any]]:
        """Diagnostics for model routing, validation failures, and fallbacks."""
        return list(self._llm_attempts)

    # -- Utility ------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            try:
                print(msg)
            except UnicodeEncodeError:
                print(str(msg).encode('ascii', errors='replace').decode('ascii'))
        logger.info(msg)
