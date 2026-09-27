"""Select a prompt-optimization method from prior experiment metadata."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

try:
    from coolprompt.utils.parsing import extract_json
except ModuleNotFoundError:
    # The selector accepts strict JSON itself, allowing lightweight use without
    # optional evaluator dependencies such as dirtyjson.
    def extract_json(text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class MetaSelectionCandidate:
    """A ranked APO configuration from the metadata base."""

    method: str
    model: str
    split: str
    metric: str
    final_score: float


@dataclass(frozen=True)
class MetaSelectionResult:
    """The auditable result of selecting an APO method."""

    profile: dict[str, str]
    similar_datasets: list[str]
    candidates: list[MetaSelectionCandidate]
    recommended_method: str
    selected_method: str
    fallback_reason: str | None
    recommended_model: str | None
    recommended_split: str | None
    recommended_metric: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form suitable for telemetry."""
        result = asdict(self)
        result["candidates"] = [asdict(item) for item in self.candidates]
        return result


class APOMetaSelector:
    """Choose RIDER or HyPER from quality, cost, and runtime metadata.

    The selector keeps method choice independent from the caller's LLM: model
    recommendations are returned for diagnostics only and never replace it.
    """

    _REQUIRED_COLUMNS = (
        "method",
        "model",
        "dataset",
        "description",
        "domain",
        "task_type",
        "split",
        "metric",
        "metric_value",
        "cost_usd",
        "time_sec",
        "llm_calls",
    )

    def __init__(self, meta_classifier_path: str | Path | None = None) -> None:
        self._path = Path(meta_classifier_path) if meta_classifier_path else None
        self._rows = self._load_rows()

    def select(
        self,
        system_model: Any,
        start_prompt: str,
        task: str,
        problem_description: str | None = None,
        dataset_name: str | None = None,
    ) -> MetaSelectionResult:
        """Profile a task, retrieve similar datasets, and rank configurations."""
        profile = self._profile_task(
            system_model, start_prompt, task, problem_description, dataset_name
        )
        similar_datasets = self._select_similar_datasets(system_model, profile)
        candidates = self._rank_candidates(profile, similar_datasets)

        if not candidates:
            return MetaSelectionResult(
                profile=profile,
                similar_datasets=similar_datasets,
                candidates=[],
                recommended_method="HyPER",
                selected_method="hyper",
                fallback_reason="No comparable metadata candidates; defaulted to HyPER.",
                recommended_model=None,
                recommended_split=None,
                recommended_metric=None,
            )

        winner = candidates[0]
        selected_method, fallback_reason = self._map_method(winner.method)
        return MetaSelectionResult(
            profile=profile,
            similar_datasets=similar_datasets,
            candidates=candidates,
            recommended_method=winner.method,
            selected_method=selected_method,
            fallback_reason=fallback_reason,
            recommended_model=winner.model,
            recommended_split=winner.split,
            recommended_metric=winner.metric,
        )

    @classmethod
    def no_dataset_result(cls) -> MetaSelectionResult:
        """Describe the documented no-data fallback for ``method='auto'``."""
        return MetaSelectionResult(
            profile={},
            similar_datasets=[],
            candidates=[],
            recommended_method="HyPER Light",
            selected_method="hyper_light",
            fallback_reason="No dataset and target were supplied; selected HyPER Light.",
            recommended_model=None,
            recommended_split=None,
            recommended_metric=None,
        )

    def _load_rows(self) -> list[dict[str, str]]:
        if self._path is not None:
            if not self._path.is_file():
                raise FileNotFoundError(f"Metadata table was not found: {self._path}")
            with self._path.open("r", encoding="utf-8-sig", newline="") as file:
                return self._normalize_rows(csv.DictReader(file))

        data_path = resources.files("coolprompt.meta_selector.data").joinpath(
            "apo_meta_base.csv"
        )
        with data_path.open("r", encoding="utf-8", newline="") as file:
            return self._normalize_rows(csv.DictReader(file))

    def _normalize_rows(self, rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        aliases = {
            "method": ("method", "метод APO"),
            "model": ("model", "llm"),
            "dataset": ("dataset", "название датасета"),
            "description": ("description", "описание задачи датасета"),
            "domain": ("domain", "домен"),
            "task_type": ("task_type", "тип задачи"),
            "split": ("split", "размер выборки (train/val/test)"),
            "metric": ("metric", "метрика"),
            "metric_value": ("metric_value", "значение метрики"),
            "cost_usd": ("cost_usd", "стоимость оптимизации в долларах"),
            "time_sec": ("time_sec", "время оптимизации (с)"),
            "llm_calls": ("llm_calls", "кол-во вызовов LLM"),
        }
        normalized = []
        for row in rows:
            item = {}
            for key, names in aliases.items():
                item[key] = next(
                    (
                        str(row.get(name, "") or "").strip()
                        for name in names
                        if row.get(name)
                    ),
                    "",
                )
            if not item["method"] or not item["dataset"]:
                continue
            item["method"] = self._canonical_method(item["method"])
            item["model"] = self._canonical_model(item["model"])
            item["dataset"] = item["dataset"].lower()
            item["domain"] = item["domain"].lower()
            item["task_type"] = item["task_type"].lower()
            normalized.append(item)
        if not normalized:
            raise ValueError("Metadata table contains no valid APO records.")
        return normalized

    def _profile_task(
        self,
        system_model: Any,
        start_prompt: str,
        task: str,
        problem_description: str | None,
        dataset_name: str | None,
    ) -> dict[str, str]:
        fallback = {
            "description": problem_description or start_prompt,
            "domain": "general",
            "task_type": task.lower(),
        }
        prompt = (
            "Return JSON only with keys description, domain, task_type. "
            "Profile this prompt-optimization task concisely.\n"
            f"Dataset name: {dataset_name or 'unknown'}\n"
            f"Declared task: {task}\n"
            f"Problem description: {problem_description or 'not provided'}\n"
            f"Initial prompt: {start_prompt}"
        )
        parsed = self._invoke_json(system_model, prompt)
        if not parsed:
            return fallback
        return {
            "description": str(parsed.get("description") or fallback["description"]),
            "domain": str(parsed.get("domain") or fallback["domain"]).lower(),
            "task_type": str(parsed.get("task_type") or fallback["task_type"]).lower(),
        }

    def _select_similar_datasets(
        self, system_model: Any, profile: dict[str, str]
    ) -> list[str]:
        catalog = {}
        for row in self._rows:
            catalog.setdefault(row["dataset"], row)
        catalog_text = "\n".join(
            f"- {name}: domain={row['domain']}; task_type={row['task_type']}; "
            f"description={row['description']}"
            for name, row in sorted(catalog.items())
        )
        prompt = (
            'Return JSON only as {"top_datasets": [..]}. Choose up to three '
            "dataset names strictly from the catalog that are most similar to the target.\n"
            f"Target: description={profile['description']}; domain={profile['domain']}; "
            f"task_type={profile['task_type']}\nCatalog:\n{catalog_text}"
        )
        parsed = self._invoke_json(system_model, prompt)
        valid = set(catalog)
        selected = []
        for item in (parsed or {}).get("top_datasets", []):
            name = str(item).strip().lower()
            if name in valid and name not in selected:
                selected.append(name)
        if selected:
            return selected[:3]

        # Deterministic fallback makes auto-selection usable with local models
        # that cannot reliably emit the requested JSON.
        scored = []
        query = " ".join(profile.values()).lower()
        for name, row in catalog.items():
            score = 0
            if row["domain"] and row["domain"] in profile["domain"]:
                score += 3
            if row["task_type"] and row["task_type"] in profile["task_type"]:
                score += 3
            score += len(
                set(re.findall(r"\w+", query))
                & set(re.findall(r"\w+", row["description"].lower()))
            )
            scored.append((score, name))
        return [name for _, name in sorted(scored, reverse=True)[:3]]

    def _rank_candidates(
        self, profile: dict[str, str], similar_datasets: list[str]
    ) -> list[MetaSelectionCandidate]:
        grouped: dict[tuple[str, str, str, str], list[tuple[float, float]]] = (
            defaultdict(list)
        )
        for dataset in similar_datasets:
            rows = [row for row in self._rows if row["dataset"] == dataset]
            if not rows:
                continue
            relevance = 0.4
            reference = rows[0]
            if reference["domain"] and reference["domain"] == profile["domain"]:
                relevance += 0.3
            if (
                reference["task_type"]
                and reference["task_type"] == profile["task_type"]
            ):
                relevance += 0.3
            for metric in {row["metric"] for row in rows}:
                candidates = [row for row in rows if row["metric"] == metric]
                quality = self._rank_scores(
                    [self._to_float(row["metric_value"]) for row in candidates], True
                )
                cost = self._rank_scores(
                    [self._to_float(row["cost_usd"]) for row in candidates], False
                )
                runtime = self._rank_scores(
                    [self._to_float(row["time_sec"]) for row in candidates], False
                )
                for index, row in enumerate(candidates):
                    score = (quality[index] + cost[index] + runtime[index]) / 3
                    key = (row["method"], row["model"], row["split"], row["metric"])
                    grouped[key].append((score, relevance))

        result = []
        for (method, model, split, metric), scores in grouped.items():
            denominator = sum(weight for _, weight in scores)
            final_score = sum(score * weight for score, weight in scores) / denominator
            result.append(
                MetaSelectionCandidate(method, model, split, metric, final_score)
            )
        return sorted(result, key=lambda candidate: candidate.final_score, reverse=True)

    @staticmethod
    def _rank_scores(values: list[float | None], larger_is_better: bool) -> list[float]:
        if len(values) == 1:
            return [1.0]
        ordered = sorted(
            range(len(values)),
            key=lambda index: (
                values[index] is None,
                -(values[index] or 0.0) if larger_is_better else values[index] or 0.0,
            ),
        )
        scores = [0.0] * len(values)
        for rank, index in enumerate(ordered):
            scores[index] = 1.0 - rank / (len(values) - 1)
        return scores

    @staticmethod
    def _invoke_json(system_model: Any, prompt: str) -> dict[str, Any] | None:
        try:
            response = system_model.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = extract_json(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            return float(value.replace(",", "."))
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _canonical_method(value: str) -> str:
        lowered = value.lower()
        if "sapo" in lowered:
            return "SAPO"
        if "rider" in lowered:
            return "RIDER"
        if "hyper" in lowered:
            return "HyPER"
        return value

    @staticmethod
    def _canonical_model(value: str) -> str:
        lowered = value.lower()
        if lowered in {"gpt-4o-mini", "openai/gpt-4o-mini"}:
            return "openai/gpt-4o-mini"
        if lowered in {"gpt-5.4-nano", "openai/gpt-5.4-nano"}:
            return "openai/gpt-5.4-nano"
        return value

    @staticmethod
    def _map_method(method: str) -> tuple[str, str | None]:
        if method == "RIDER":
            return "rider", None
        if method == "HyPER":
            return "hyper", None
        if method == "SAPO":
            return "hyper", "SAPO is not implemented in CoolPrompt"
        return "hyper", f"Unsupported recommendation '{method}'; defaulted to HyPER."
