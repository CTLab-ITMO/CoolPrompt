from dataclasses import dataclass, field
from typing import List


@dataclass
class OptimizerState:
    """Compact normalized state for the controller.

    All scalar fields should be in [0, 1] whenever possible.
    """

    val_quality: float = 0.0
    quality_slope: float = 0.0
    stagnation: float = 0.0
    useless_ops_ratio: float = 0.0
    remaining_budget_ratio: float = 1.0
    epoch_progress: float = 0.0
    population_diversity: float = 0.5


@dataclass
class ReflectionRecord:
    text: str
    source_action: str
    utility_score: float = 0.0
    contradiction_score: float = 0.0
    tags: List[str] = field(default_factory=list)
