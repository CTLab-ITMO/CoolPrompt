"""Hyperparameters of the original RIDER (FRUCT paper version)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Operator classes and their share of the initial budget (paper, Sec. III-C).
OPERATOR_CLASSES: dict[str, tuple[float, tuple[str, ...]]] = {
    "eda": (0.42, ("eda_mutation", "eda_rank_index")),
    "generative": (0.31, ("zero_order", "first_order_refinement")),
    "evolutionary": (0.20, ("ga_mutation", "ga_crossover", "de_mutation")),
    "reflective": (0.07, ("reflection_crossover", "lamarckian")),
}

OPERATORS: tuple[str, ...] = tuple(
    name for _, names in OPERATOR_CLASSES.values() for name in names
)

# Reflective operators need a single correct output to reason about errors, so
# they are disabled for open-ended generation (XSum, CommonGen in the paper).
OPEN_ENDED_DISABLED_OPERATORS: frozenset[str] = frozenset(
    {"reflection_crossover", "lamarckian"}
)
# Metric class-name fragments that mean "no single correct output".
OPEN_ENDED_METRICS: tuple[str, ...] = ("bert", "rouge", "bleu", "meteor", "judge", "geval")


def operator_budget_shares() -> dict[str, float]:
    """Per-operator share of the initial budget (class share split evenly)."""

    shares: dict[str, float] = {}
    for share, names in OPERATOR_CLASSES.values():
        for name in names:
            shares[name] = share / len(names)
    return shares


@dataclass
class RIDERConfig:
    """All knobs of the evolutionary loop; defaults follow the FRUCT paper."""

    population_size: int = 12
    num_generations: int = 8
    offspring_multiplier: int = 3
    elite_size: int = 3
    tournament_size: int = 3

    # Initialization channels (data-aware + zero-order + fallback = N).
    data_aware_seeds: int = 5
    zero_order_seeds: int = 5
    fallback_seeds: int = 2
    data_aware_examples: int = 5
    data_aware_temperature: tuple[float, float] = (0.8, 1.2)
    zero_order_temperature: tuple[float, float] = (0.5, 1.3)
    min_prompt_words: int = 15
    regeneration_attempts: int = 3

    # Adaptive operator selection.
    base_temperature: float = 0.7
    epsilon: float = 0.10
    stagnation_epsilon: float = 0.25
    thompson_window: int = 5
    thompson_decay: float = 0.9
    prior_strength: float = 4.0
    penalty_reward_threshold: float = 0.3
    penalty_min_trials: int = 10

    # Evaluate-first (mu + lambda) retention.
    diversity_threshold: float = 0.92
    fitness_tie_tolerance: float = 0.01

    # Stagnation escape.
    stagnation_improvement: float = 0.01
    stagnation_generations: int = 2
    restart_generations: int = 4
    stagnation_temperature_factor: float = 1.3
    recovery_temperature_factor: float = 0.9
    max_temperature: float = 1.5
    restart_fraction: float = 0.5

    # Error-directed evolution and long-term memory.
    num_error_examples: int = 5
    memory_update_interval: int = 3
    memory_patterns: int = 3

    # Output ensemble.
    ensemble_size: int = 5
    ensemble_quality_weight: float = 1.0

    # Stop early once a prompt reaches the metric's maximum; nothing can improve it.
    stop_at_perfect: bool = True
    perfect_score: float = 1.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_workers: int = 8
    seed: int | None = 42
    disabled_operators: frozenset[str] = field(default_factory=frozenset)

    @property
    def offspring_per_generation(self) -> int:
        return self.offspring_multiplier * self.population_size

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.num_generations < 0:
            raise ValueError("num_generations must be non-negative")
        if self.offspring_multiplier < 1:
            raise ValueError("offspring_multiplier must be at least 1")
        if not 0.0 <= self.epsilon <= 1.0 or not 0.0 <= self.stagnation_epsilon <= 1.0:
            raise ValueError("epsilon values must lie in [0, 1]")
        if not 0.0 < self.thompson_decay <= 1.0:
            raise ValueError("thompson_decay must lie in (0, 1]")
        unknown = set(self.disabled_operators) - set(OPERATORS)
        if unknown:
            raise ValueError(f"Unknown operators: {sorted(unknown)}")
