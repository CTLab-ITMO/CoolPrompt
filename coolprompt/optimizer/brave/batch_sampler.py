from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from coolprompt.utils.enums import Task


@dataclass(frozen=True)
class GenerationFeatures:
    input_len: int
    target_len: int


class StratifiedBatchSampler:
    """Builds balanced train batches for every optimization epoch."""

    def __init__(
        self,
        task: Task,
        batch_size: int,
        seed: int = 19,
        generation_bins: int = 3,
    ) -> None:
        self.task = task
        self.batch_size = max(int(batch_size), 1)
        self.generation_bins = max(int(generation_bins), 2)
        self._seed = int(seed)

    def _build_generation_features(
        self,
        dataset: List[str],
        targets: List[str],
    ) -> List[GenerationFeatures]:
        features: List[GenerationFeatures] = []
        for source, target in zip(dataset, targets):
            features.append(
                GenerationFeatures(
                    input_len=len(source),
                    target_len=len(target),
                )
            )
        return features

    def _quantile_edges(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.array([], dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, self.generation_bins + 1)[1:-1]
        if quantiles.size == 0:
            return np.array([], dtype=np.float64)
        edges = np.quantile(values, quantiles)
        return np.unique(edges.astype(np.float64))

    @staticmethod
    def _assign_bin(value: float, edges: np.ndarray) -> int:
        if edges.size == 0:
            return 0
        return int(np.searchsorted(edges, value, side="right"))

    def _build_strata(
        self,
        dataset: List[str],
        targets: List[str | int],
    ) -> Dict[Tuple[str, ...], List[int]]:
        strata: Dict[Tuple[str, ...], List[int]] = {}
        if self.task == Task.CLASSIFICATION:
            for idx, target in enumerate(targets):
                key = (str(target),)
                strata.setdefault(key, []).append(idx)
            return strata

        features = self._build_generation_features(dataset, targets)
        input_lengths = np.array(
            [f.input_len for f in features],
            dtype=np.float64
        )
        target_lengths = np.array(
            [f.target_len for f in features],
            dtype=np.float64
        )
        input_edges = self._quantile_edges(input_lengths)
        target_edges = self._quantile_edges(target_lengths)

        for idx, feats in enumerate(features):
            key = (
                str(self._assign_bin(feats.input_len, input_edges)),
                str(self._assign_bin(feats.target_len, target_edges)),
            )
            strata.setdefault(key, []).append(idx)
        return strata

    def _compute_quotas(
        self,
        strata_sizes: Dict[Tuple[str, ...], int],
        total_size: int,
    ) -> Dict[Tuple[str, ...], int]:
        if total_size <= 0:
            return {}
        target_size = min(self.batch_size, total_size)
        expected = {
            key: target_size * (size / total_size)
            for key, size in strata_sizes.items()
        }
        base = {key: int(np.floor(value)) for key, value in expected.items()}
        assigned = sum(base.values())
        remainder = target_size - assigned

        if remainder > 0:
            ranked = sorted(
                expected.items(),
                key=lambda item: item[1] - base[item[0]],
                reverse=True
            )
            for key, _ in ranked[:remainder]:
                base[key] += 1
        return base

    def sample(
        self,
        dataset: Sequence[str],
        targets: Sequence[str | int],
        epoch: int,
    ) -> List[int]:
        total_size = len(dataset)
        if total_size == 0:
            return []
        if total_size <= self.batch_size:
            return list(range(total_size))

        rng = np.random.default_rng(self._seed + int(epoch))
        strata = self._build_strata(dataset, targets)
        strata_sizes = {k: len(v) for k, v in strata.items()}
        quotas = self._compute_quotas(strata_sizes, total_size)

        selected: List[int] = []
        selected_set = set()
        leftovers: List[int] = []
        for key, indices in strata.items():
            quota = quotas.get(key, 0)
            if quota <= 0:
                leftovers.extend(indices)
                continue
            shuffled = list(indices)
            rng.shuffle(shuffled)
            take = min(quota, len(shuffled))
            picked = shuffled[:take]
            selected.extend(picked)
            selected_set.update(picked)
            leftovers.extend(shuffled[take:])

        if len(selected) < self.batch_size:
            remain = [idx for idx in leftovers if idx not in selected_set]
            rng.shuffle(remain)
            need = self.batch_size - len(selected)
            selected.extend(remain[:need])

        if len(selected) < self.batch_size:
            need = self.batch_size - len(selected)
            fallback = rng.choice(total_size, size=need, replace=True).tolist()
            selected.extend(int(x) for x in fallback)

        rng.shuffle(selected)
        return selected[:self.batch_size]


class CurriculumStratifiedBatchSampler(StratifiedBatchSampler):
    """Stratified sampler that gradually up-weights hard examples.

    During warmup_steps: pure stratified sampling (alpha=0).
    After warmup: within each stratum, example weights blend uniform
    and difficulty-based as alpha linearly grows to max_alpha.
    """

    def __init__(
        self,
        task: Task,
        batch_size: int,
        total_steps: int,
        seed: int = 19,
        generation_bins: int = 3,
        warmup_steps: int = 20,
        max_alpha: float = 0.6,
    ) -> None:
        super().__init__(task, batch_size, seed, generation_bins)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.max_alpha = float(np.clip(max_alpha, 0.0, 1.0))
        self._error_counts: Dict[int, int] = {}
        self._eval_counts: Dict[int, int] = {}

    def update_difficulties(
        self,
        batch_indices: List[int],
        failed_indices: List[int],
    ) -> None:
        """Record per-example outcomes after an evaluation step.

        Args:
            batch_indices: global dataset indices that were in the batch.
            failed_indices: subset of batch_indices where the prompt failed.
        """
        failed_set = set(failed_indices)
        for idx in batch_indices:
            self._eval_counts[idx] = self._eval_counts.get(idx, 0) + 1
            if idx in failed_set:
                self._error_counts[idx] = self._error_counts.get(idx, 0) + 1

    def _curriculum_alpha(self, epoch: int) -> float:
        if epoch <= self.warmup_steps:
            return 0.0
        ramp_len = max(self.total_steps - self.warmup_steps, 1)
        return self.max_alpha * min((epoch - self.warmup_steps) / ramp_len, 1.0)

    def _difficulty(self, idx: int) -> float:
        evals = self._eval_counts.get(idx, 0)
        if evals == 0:
            return 0.5  # neutral prior for unseen examples
        return self._error_counts.get(idx, 0) / evals

    def sample(
        self,
        dataset: Sequence[str],
        targets: Sequence[str | int],
        epoch: int,
    ) -> List[int]:
        alpha = self._curriculum_alpha(epoch)
        if alpha == 0.0:
            return super().sample(dataset, targets, epoch)

        total_size = len(dataset)
        if total_size == 0:
            return []
        if total_size <= self.batch_size:
            return list(range(total_size))

        rng = np.random.default_rng(self._seed + int(epoch))
        strata = self._build_strata(dataset, targets)
        strata_sizes = {k: len(v) for k, v in strata.items()}
        quotas = self._compute_quotas(strata_sizes, total_size)

        selected: List[int] = []
        leftovers: List[int] = []

        for key, indices in strata.items():
            quota = quotas.get(key, 0)
            if quota <= 0:
                leftovers.extend(indices)
                continue

            take = min(quota, len(indices))
            if take >= len(indices):
                selected.extend(indices)
                continue

            # Within-stratum blend: uniform + difficulty
            raw = np.array(
                [(1.0 - alpha) + alpha * self._difficulty(i) for i in indices],
                dtype=np.float64,
            )
            raw = np.clip(raw, 1e-12, None)
            p = raw / raw.sum()

            picked_local = rng.choice(
                len(indices), size=take, replace=False, p=p
            ).tolist()
            picked_set = set(picked_local)
            selected.extend(indices[i] for i in picked_local)
            leftovers.extend(
                indices[i] for i in range(len(indices)) if i not in picked_set
            )

        if len(selected) < self.batch_size:
            selected_set = set(selected)
            remain = [idx for idx in leftovers if idx not in selected_set]
            rng.shuffle(remain)
            selected.extend(remain[:self.batch_size - len(selected)])

        if len(selected) < self.batch_size:
            need = self.batch_size - len(selected)
            fallback = rng.choice(total_size, size=need, replace=True).tolist()
            selected.extend(int(x) for x in fallback)

        rng.shuffle(selected)
        return selected[:self.batch_size]
