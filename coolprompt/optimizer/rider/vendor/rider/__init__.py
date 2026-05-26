"""
RIDER - Reflective Iterative Diversity-Enhanced Reasoning.

A metaheuristic algorithm for automatic prompt optimization using evolutionary computation.
"""

__version__ = '0.1.0'
__author__ = 'RIDER Research Team'

from rider.algorithms.rider import RIDER
from rider.core.prompts import Prompt
from rider.config.schema import ExperimentConfig, load_config_from_yaml
from rider.assistant import RiderGenesis

__all__ = ['RIDER', 'Prompt', 'ExperimentConfig', 'load_config_from_yaml', 'RiderGenesis']
