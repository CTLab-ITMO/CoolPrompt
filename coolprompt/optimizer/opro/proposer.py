"""OPRO proposer: trajectory-based prompt optimization."""

import random
from typing import List, Optional

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
)
from coolprompt.utils.prompt_templates.opro_templates import (
    OPRO_META_TEMPLATE,
)


class OPROProposer:
    """Generates prompts based on a trajectory of past attempts.

    Maintains a sorted history of (prompt, score) pairs and
    asks the LLM to propose a prompt that scores higher than
    all previous attempts. Includes task demonstrations from
    training data in the meta-prompt, matching the original
    OPRO paper.

    Args:
        model (BaseLanguageModel): LLM for meta-optimization.
        train_dataset (List[str]): Training input samples.
        train_targets (List[str]): Training ground-truth
            targets.
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        max_trajectory (int): Maximum number of past attempts
            to keep in the trajectory.
        n_demonstrations (int): Number of task demonstrations
            to include in the meta-prompt.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        train_dataset: List[str],
        train_targets: List[str],
        prompt_max_tokens: int = 300,
        max_trajectory: int = 20,
        n_demonstrations: int = 5,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.prompt_max_tokens = prompt_max_tokens
        self.max_trajectory = max_trajectory
        self.n_demonstrations = n_demonstrations
        self.trajectory: list[tuple[str, float]] = []

    def update_trajectory(
        self, prompt: str, score: float
    ) -> None:
        """Adds a (prompt, score) pair to the trajectory.

        Keeps only the top max_trajectory entries by score.

        Args:
            prompt (str): The prompt text.
            score (float): The evaluation score.
        """
        self.trajectory.append((prompt, score))
        self.trajectory.sort(key=lambda x: x[1])
        if len(self.trajectory) > self.max_trajectory:
            self.trajectory = self.trajectory[
                -self.max_trajectory:
            ]

    def _format_trajectory(self) -> str:
        """Formats trajectory as a numbered list, worst to best.

        Returns:
            str: Formatted trajectory string.
        """
        parts = []
        for i, (prompt, score) in enumerate(
            self.trajectory, 1
        ):
            parts.append(
                f"{i}. Score: {score:.4f}\n"
                f"   Instruction: {prompt}"
            )
        return "\n\n".join(parts)

    def _format_task_demonstrations(self) -> str:
        """Samples and formats task input-output pairs.

        Returns:
            str: Formatted task demonstrations string.
        """
        n = min(
            self.n_demonstrations, len(self.train_dataset)
        )
        indices = random.sample(
            range(len(self.train_dataset)), n
        )
        parts = []
        for i, idx in enumerate(indices, 1):
            parts.append(
                f"{i}. Input: {self.train_dataset[idx]}\n"
                f"   Output: {self.train_targets[idx]}"
            )
        return "\n\n".join(parts)

    def propose(self) -> tuple[str, str]:
        """Proposes a prompt based on the trajectory.

        Returns:
            tuple[str, str]: (new_prompt, "trajectory").
        """
        trajectory_str = self._format_trajectory()
        demos_str = self._format_task_demonstrations()

        prompt = OPRO_META_TEMPLATE.format(
            task_demonstrations=demos_str,
            trajectory=trajectory_str,
            max_tokens=self.prompt_max_tokens,
        )
        answer = get_model_answer_extracted(
            self.model, prompt
        )

        fallback = (
            self.trajectory[-1][0]
            if self.trajectory
            else ""
        )
        new_prompt = extract_answer(
            answer,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=fallback,
        )

        return new_prompt.strip(), "trajectory"
