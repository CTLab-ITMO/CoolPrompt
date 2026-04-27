"""Legacy-compatible RIDER optimizer facade.

This module wraps the **unmodified** RiderGenesis implementation vendored
from the rider repository (``coolprompt.optimizer.rider._rider_assistant``)
so that LIGHT mode runs byte-for-byte identically to the upstream
``RiderGenesis.run_light`` flow — same prompt templates, same control
flow, same temperatures, same parsing — while routing all LLM traffic
through a langchain ``BaseLanguageModel`` instead of OpenRouter HTTP.

The PR #88 ``AutoPromptingMethod`` interface is not yet merged into
``stage``; once it lands, an additional wrapper class will be added.
"""

from __future__ import annotations

from typing import Optional

from coolprompt.optimizer.hype.hype import Optimizer
from coolprompt.optimizer.rider import _llm_shim
from coolprompt.optimizer.rider._rider_assistant import RiderGenesis


class RIDEROptimizer(Optimizer):
    """RIDER LIGHT prompt optimizer.

    LIGHT mode runs the upstream ``RiderGenesis.run_light`` pipeline with
    no behavioral modifications. The five LLM calls are:

    1. Contract extraction (planning model, T=0.7)
    2. Strategy A (working model, IGNITION T=1.15) — picked adaptively
    3. Strategy B (working model, IGNITION T=1.15) — picked adaptively
    4. Pairwise comparison vs original (planning model, T=0.0)
    5. Quality estimation (planning model, T=0.0)
    """

    def __init__(
        self,
        model,
        planning_model=None,
        mode: str = "light",
        verbose: bool = False,
    ) -> None:
        """Args:
            model: langchain ``BaseLanguageModel`` for the working model.
            planning_model: langchain ``BaseLanguageModel`` for planning
                steps (contract extraction, pairwise compare, quality
                estimate). Defaults to ``model`` if omitted.
            mode: only ``"light"`` is currently implemented.
            verbose: passed to RiderGenesis (controls stdout logging).

        Raises:
            NotImplementedError: if ``mode`` is anything other than
                ``"light"``.
        """
        super().__init__(model)
        if mode != "light":
            raise NotImplementedError(
                "Only RIDER light mode is wired up in this scaffold. "
                "blitz, standard, and ultra modes are reserved for future "
                "integration."
            )
        self.planning_model = planning_model or model
        self.mode = mode
        self.verbose = verbose
        self._rider: Optional[RiderGenesis] = None

    def _ensure_rider(self) -> RiderGenesis:
        if self._rider is None:
            # Register langchain models BEFORE constructing RiderGenesis,
            # because its __init__ creates an LLMClient instance that
            # snapshots the current registry on each call.
            _llm_shim.register_models(self.model, self.planning_model)
            # ``api_key="-"`` satisfies the upstream init guard; the shim
            # ignores both provider and api_key.
            self._rider = RiderGenesis(
                model="working",  # any non-PLANNING_MODEL string routes to working
                api_key="-",
                verbose=self.verbose,
            )
        else:
            # Refresh registry in case the caller swapped models between
            # optimize() invocations.
            _llm_shim.register_models(self.model, self.planning_model)
        return self._rider

    def optimize(self, prompt: str) -> str:
        """Return an improved prompt via the upstream RIDER LIGHT flow."""
        rider = self._ensure_rider()
        return rider.run_light(prompt)

    @property
    def api_calls(self) -> int:
        """Total LLM calls made by the most recent ``optimize`` invocation."""
        if self._rider is None:
            return 0
        return self._rider.api_calls
