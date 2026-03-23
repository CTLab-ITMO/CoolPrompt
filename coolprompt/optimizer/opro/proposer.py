"""OPRO proposer: trajectory-based prompt optimization."""

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.pe2.node import Node
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
    all previous attempts.

    Args:
        model (BaseLanguageModel): LLM for meta-optimization.
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        max_trajectory (int): Maximum number of past attempts
            to keep in the trajectory.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
        max_trajectory: int = 20,
    ) -> None:
        self.model = model
        self.prompt_max_tokens = prompt_max_tokens
        self.max_trajectory = max_trajectory
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

    def propose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
    ) -> tuple[str, str]:
        """Proposes a prompt based on the trajectory.

        Args:
            node (Node): Current beam node (used as fallback).
            examples_str (str): Ignored by OPRO.
            full_template (str): Ignored by OPRO.
            batch_size (int): Ignored by OPRO.

        Returns:
            tuple[str, str]: (new_prompt, "trajectory").
        """
        trajectory_str = self._format_trajectory()

        prompt = OPRO_META_TEMPLATE.format(
            trajectory=trajectory_str,
            max_tokens=self.prompt_max_tokens,
        )
        answer = get_model_answer_extracted(self.model, prompt)

        new_prompt = extract_answer(
            answer,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
        )

        return new_prompt.strip(), "trajectory"
