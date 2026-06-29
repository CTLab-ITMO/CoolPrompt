from abc import ABC, abstractmethod
from typing import Optional, Any
from coolprompt.optimizer.brave.operation_logger import OperationLogger


class Operator(ABC):
    def __init__(self, logger: Optional[OperationLogger] = None) -> None:
        self.logger = logger

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass
