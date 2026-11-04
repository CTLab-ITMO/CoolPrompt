"""Adaptive operator selection: epsilon-greedy over Windowed Thompson Sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class _Arm:
    prior_alpha: float
    prior_beta: float
    # (generation, reward in [0, 1]) for every application of the operator.
    outcomes: list[tuple[int, float]] = field(default_factory=list)

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    @property
    def mean_reward(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(r for _, r in self.outcomes) / len(self.outcomes)


class WindowedThompsonSelector:
    """Two-layer operator selection from the paper (Sec. III-D).

    Layer 1 picks a uniformly random operator with probability ``epsilon``.
    Layer 2 samples ``theta ~ Beta(alpha, beta)`` per operator and takes the
    argmax. When choosing for generation ``g`` the posterior uses rewards from
    generations ``g - window`` .. ``g - 1``, discounted by ``decay`` per extra
    generation of age, so early failures do not suppress an operator forever.
    An operator whose average reward inside that window stays below
    ``penalty_threshold`` over at least ``penalty_min_trials`` applications gets
    the soft penalty ``alpha <- 0.3 alpha, beta <- max(beta, 10)``; it lifts as
    soon as the operator does better or its failures leave the window.
    """

    def __init__(
        self,
        operators: list[str],
        prior_means: dict[str, float],
        *,
        prior_strength: float,
        window: int,
        decay: float,
        penalty_threshold: float,
        penalty_min_trials: int,
        rng: random.Random,
    ) -> None:
        if not operators:
            raise ValueError("At least one operator is required")
        self.window = window
        self.decay = decay
        self.penalty_threshold = penalty_threshold
        self.penalty_min_trials = penalty_min_trials
        self.rng = rng
        self.arms: dict[str, _Arm] = {}
        for name in operators:
            mean = min(max(prior_means.get(name, 0.5), 0.01), 0.99)
            self.arms[name] = _Arm(
                prior_alpha=1.0 + prior_strength * mean,
                prior_beta=1.0 + prior_strength * (1.0 - mean),
            )

    def _windowed(self, name: str, generation: int) -> list[tuple[int, float]]:
        return [
            (generation - gen, reward)
            for gen, reward in self.arms[name].outcomes
            if 1 <= generation - gen <= self.window
        ]

    def posterior(self, name: str, generation: int) -> tuple[float, float]:
        arm = self.arms[name]
        alpha, beta = arm.prior_alpha, arm.prior_beta
        recent = self._windowed(name, generation)
        for age, reward in recent:
            weight = self.decay ** (age - 1)
            alpha += weight * reward
            beta += weight * (1.0 - reward)
        if len(recent) >= self.penalty_min_trials and (
            sum(r for _, r in recent) / len(recent) < self.penalty_threshold
        ):
            alpha = 0.3 * alpha
            beta = max(beta, 10.0)
        return alpha, beta

    def select(self, generation: int, epsilon: float) -> str:
        names = list(self.arms)
        if self.rng.random() < epsilon:
            return self.rng.choice(names)
        best_name, best_theta = names[0], -1.0
        for name in names:
            alpha, beta = self.posterior(name, generation)
            theta = self.rng.betavariate(alpha, beta)
            if theta > best_theta:
                best_name, best_theta = name, theta
        return best_name

    def update(self, name: str, generation: int, reward: float) -> None:
        self.arms[name].outcomes.append((generation, min(max(reward, 0.0), 1.0)))

    def selection_probabilities(
        self, generation: int, epsilon: float, samples: int = 2000
    ) -> dict[str, float]:
        """Monte-Carlo estimate of the selection distribution, for logging."""

        counts = dict.fromkeys(self.arms, 0)
        state = self.rng.getstate()
        for _ in range(samples):
            counts[self.select(generation, epsilon)] += 1
        self.rng.setstate(state)
        return {name: count / samples for name, count in counts.items()}

    def statistics(self) -> dict[str, dict[str, float]]:
        return {
            name: {"trials": arm.trials, "mean_reward": arm.mean_reward}
            for name, arm in self.arms.items()
        }
