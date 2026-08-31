from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from coolprompt.optimizer.brave.core_states import OptimizerState


@dataclass
class ActionResult:
    """Describe the outcome and token cost of an optimizer action."""

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
        train_data: Any,
        val_data: Any,
    ) -> ActionResult:
        """Execute an action against the current optimizer context.

        Args:
            action (str): name of the action to execute.
            population (List[str]): current prompt population.
            state (OptimizerState): current normalized optimizer state.
            train_data (Any): training data available to the action.
            val_data (Any): validation data available to the action.

        Returns:
            ActionResult: measured action outcome and its payload.
        """

        pass
