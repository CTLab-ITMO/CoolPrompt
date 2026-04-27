"""RIDER optimizer package.

Public exports:
- :class:`RIDEROptimizer` — langchain-friendly facade (use this).
- :class:`RiderGenesis` — vendored upstream class for advanced users.
"""

from coolprompt.optimizer.rider._rider_assistant import RiderGenesis
from coolprompt.optimizer.rider.rider import RIDEROptimizer

__all__ = ["RIDEROptimizer", "RiderGenesis"]
