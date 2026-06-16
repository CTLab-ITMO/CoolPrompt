"""Utilities for persisting a PromptBreeder run to disk.

The :class:`RunLogger` captures three things:

* the initial population (prompts produced by ``init_run``),
* a snapshot of the population after every generation of the genetic
  algorithm, and
* the final population together with the best-overall prompt found
  across all elites.

The result is written as a single JSON document under ``runs/``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)


def _serialize_unit(unit: EvolutionUnit, index: int) -> Dict[str, Any]:
    data = unit.model_dump()
    data["unit_index"] = index
    return data


def _snapshot_units(population: Population) -> List[Dict[str, Any]]:
    return [_serialize_unit(u, i) for i, u in enumerate(population.units)]


def _best_unit(units: List[EvolutionUnit]) -> Optional[EvolutionUnit]:
    if not units:
        return None
    return max(units, key=lambda u: u.fitness)


class RunLogger:
    """Collects the initial prompt, optimization process, and final prompt."""

    def __init__(
        self,
        metadata: Dict[str, Any],
        output_dir: str = "runs",
        filename: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.filename = filename or f"run_{timestamp}.json"
        self.path = os.path.join(self.output_dir, self.filename)

        self._data: Dict[str, Any] = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **metadata,
            },
            "initial_population": [],
            "generations": [],
            "final_population": [],
            "final_best_prompt": None,
        }
        # Track the best prompt seen across all generations.
        self._best_overall: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ logging

    def log_initial(self, population: Population) -> None:
        self._data["initial_population"] = _snapshot_units(population)
        self._update_best(generation=0, population=population)

    def log_generation(self, generation_index: int, population: Population) -> None:
        elite = _best_unit(population.units)
        entry = {
            "generation": generation_index,
            "units": _snapshot_units(population),
            "elite": (
                {
                    "unit_index": population.units.index(elite),
                    "T": elite.T,
                    "M": elite.M,
                    "P": elite.P,
                    "fitness": elite.fitness,
                }
                if elite is not None
                else None
            ),
        }
        self._data["generations"].append(entry)
        self._update_best(generation=generation_index, population=population)

    def log_final(self, population: Population) -> None:
        self._data["final_population"] = _snapshot_units(population)
        # Re-scan elites list as well in case best lived only briefly.
        for elite in population.elites:
            if (
                self._best_overall is None
                or elite.fitness > self._best_overall["fitness"]
            ):
                self._best_overall = {
                    "T": elite.T,
                    "M": elite.M,
                    "P": elite.P,
                    "fitness": elite.fitness,
                    "found_at_generation": self._best_overall.get(
                        "found_at_generation"
                    )
                    if self._best_overall
                    else None,
                }
        # Annotate the final best prompt with dataset / metric for traceability.
        if self._best_overall is not None:
            meta = self._data.get("metadata", {})
            self._best_overall["dataset"] = meta.get("dataset")
            self._best_overall["metric"] = meta.get("metric")
        self._data["final_best_prompt"] = self._best_overall

    # -------------------------------------------------------------------- save

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info("Run log written to %s", target)
        return target

    # ----------------------------------------------------------------- helpers

    def _update_best(self, generation: int, population: Population) -> None:
        elite = _best_unit(population.units)
        if elite is None:
            return
        if (
            self._best_overall is None
            or elite.fitness > self._best_overall["fitness"]
        ):
            self._best_overall = {
                "T": elite.T,
                "M": elite.M,
                "P": elite.P,
                "fitness": elite.fitness,
                "found_at_generation": generation,
            }

    @property
    def data(self) -> Dict[str, Any]:
        """Read-only access to the current in-memory document."""
        return self._data
