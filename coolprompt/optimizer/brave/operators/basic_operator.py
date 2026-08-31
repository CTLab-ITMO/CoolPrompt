from abc import ABC, abstractmethod
from typing import Optional, Any
from coolprompt.optimizer.brave.operation_logger import OperationLogger


class Operator(ABC):
    """Base interface for BRAVE prompt-transformation operators."""

    def __init__(self, logger: Optional[OperationLogger] = None) -> None:
        """Store an optional operation logger.

        Args:
            logger (Optional[OperationLogger]): logger for operator
                diagnostics.
        """

        self.logger = logger

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the operator and return its generated prompt or result.

        Args:
            *args (Any): positional operator inputs.
            **kwargs (Any): keyword operator inputs.

        Returns:
            Any: operator-specific result.
        """

        pass
