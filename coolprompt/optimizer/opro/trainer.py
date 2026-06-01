"""OPRO trainer: flat iterative loop for prompt optimization."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.opro.proposer import OPROProposer
from coolprompt.utils.logging_config import logger


class OPROTrainer:
    """Standalone OPRO trainer using a flat iterative loop.

    Matches the original OPRO paper: maintains a trajectory
    of (prompt, score) pairs, proposes new candidates each
    step, evaluates on training data, and returns the best.

    Args:
        model (BaseLanguageModel): LLM used for inference.
        evaluator (Evaluator): Evaluator for scoring prompts.
        proposer (OPROProposer): Proposer for generating
            new prompts from the trajectory.
        train_dataset (List[str]): Training input samples.
        train_targets (List[str]): Training ground-truth
            targets.
        val_dataset (List[str]): Validation input samples.
        val_targets (List[str]): Validation ground-truth
            targets.
        template (str): Prompt template for the task.
        train_steps (int): Number of optimization iterations.
        n_candidates (int): Candidates generated per step.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        proposer: OPROProposer,
        train_dataset: List[str],
        train_targets: List[str],
        val_dataset: List[str],
        val_targets: List[str],
        template: str,
        train_steps: int = 3,
        n_candidates: int = 8,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.proposer = proposer
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.val_dataset = val_dataset
        self.val_targets = val_targets
        self.template = template
        self.train_steps = train_steps
        self.n_candidates = n_candidates

    def _evaluate_on_train(self, prompt: str) -> float:
        """Evaluates a prompt on training data.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            float: The evaluation score.
        """
        return self.evaluator.evaluate(
            prompt=prompt,
            dataset=self.train_dataset,
            targets=self.train_targets,
            template=self.template,
        )

    def _evaluate_on_val(self, prompt: str) -> float:
        """Evaluates a prompt on validation data.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            float: The evaluation score.
        """
        return self.evaluator.evaluate(
            prompt=prompt,
            dataset=self.val_dataset,
            targets=self.val_targets,
            template=self.template,
        )

    def train(self, initial_prompt: str) -> str:
        """Runs the OPRO iterative optimization.

        Args:
            initial_prompt (str): The starting prompt.

        Returns:
            str: The best prompt found during optimization.
        """
        score = self._evaluate_on_train(initial_prompt)
        self.proposer.update_trajectory(initial_prompt, score)
        logger.info(
            f"OPRO initial train score: {score:.4f}"
        )

        seen_prompts = {initial_prompt.strip().lower()}
        best_prompt = initial_prompt
        best_val_score = -1.0

        for t in range(self.train_steps):
            logger.info(
                f"OPRO step {t + 1}/{self.train_steps}"
            )

            def _do_propose(_):
                prompt, _ = self.proposer.propose()
                return prompt

            new_prompts = []
            with ThreadPoolExecutor(
                max_workers=min(self.n_candidates, 12)
            ) as pool:
                futures = [
                    pool.submit(_do_propose, i)
                    for i in range(self.n_candidates)
                ]
                for fut in as_completed(futures):
                    new_prompts.append(fut.result())

            unique_prompts = []
            for p in new_prompts:
                key = p.strip().lower()
                if key not in seen_prompts:
                    seen_prompts.add(key)
                    unique_prompts.append(p)

            if not unique_prompts:
                logger.info(
                    "  No new unique prompts, stopping"
                )
                break

            for p in unique_prompts:
                s = self._evaluate_on_train(p)
                self.proposer.update_trajectory(p, s)
                logger.info(
                    f"  Candidate train score: {s:.4f}"
                )

            logger.info(
                f"  Added {len(unique_prompts)} new "
                f"prompts to trajectory"
            )

        all_prompts = [
            p for p, _ in self.proposer.trajectory
        ]
        for p in all_prompts:
            val_score = self._evaluate_on_val(p)
            if val_score > best_val_score:
                best_val_score = val_score
                best_prompt = p

        logger.info(
            f"OPRO best prompt "
            f"(val={best_val_score:.4f})"
        )
        return best_prompt
