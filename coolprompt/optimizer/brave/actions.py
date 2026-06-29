from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from coolprompt.optimizer.begrape.core_states import (
    OptimizerState, EpistemicMemory
)


@dataclass
class ActionResult:
    action: str
    delta_quality: float
    cost_tokens: float
    payload: Dict[str, Any] = field(default_factory=dict)
    improved: bool = False


class ActionExecutor(Protocol):
    """Executor interface for domain-specific implementation.

    You can implement this against your existing GRAPE pipeline.
    """

    def execute(
        self,
        action: str,
        population: List[str],
        state: OptimizerState,
        memory: EpistemicMemory,
        train_data: Any,
        val_data: Any,
    ) -> ActionResult:
        pass
