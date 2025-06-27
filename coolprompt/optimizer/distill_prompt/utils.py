"""Utility functions for prompt optimization experiments.

This module provides a TextSampler class for drawing random examples from a
dataset and a utility function to set random seeds for major libraries to
ensure experimental reproducibility.
"""

import os
import random
from typing import List, Tuple

import numpy as np
import torch


class TextSampler:
    """A simple class to randomly sample text-label pairs from a dataset."""

    def __init__(self, texts: List[str], labels: List[str]) -> None:
        """Initializes the TextSampler with texts and corresponding labels.

        Args:
            texts (List[str]): A list of text strings.
            labels (List[str]): A list of corresponding labels.
        """
        self.texts = texts
        self.labels = labels

    def sample(self, count: int) -> List[Tuple[str, str]]:
        """Samples a specified number of text-label pairs without replacement.

        If the requested count is larger than the dataset size, it returns
        all the available data.

        Args:
            count (int): The number of samples to retrieve.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains
                a text and its corresponding label.
        """
        sample_size = min(count, len(self.texts))
        indices = random.sample(range(len(self.texts)), sample_size)
        return [(self.texts[i], self.labels[i]) for i in indices]


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch to ensure
    reproducibility.

    This function sets seeds for the `random` module, `os.environ`,
    `numpy`, and `torch` (if available).

    Args:
        seed (int, optional): The integer value to use for all random seeds.
            Defaults to 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
