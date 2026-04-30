from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.hype.hyper import HyPEROptimizer
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.enums import Task


class HyPERMethod(AutoPromptingMethod):
    """HyPER (Hypothesis‑Prompt Evolution with Refinement) method.

    Extends HyPE with iterative refinement: candidates are generated,
    evaluated on mini‑batches, and feedback (recommendations) is used
    to update the meta‑prompt until convergence or patience is reached.

    This method **requires** a dataset split and an evaluator.
    """

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        """Run HyPER prompt optimization.

        Args:
            model: The language model used for generation.
            initial_prompt (str): The starting prompt.
            dataset_split (tuple): A 4‑tuple of
                (train_samples, val_samples, train_targets, val_targets).
            evaluator (Evaluator): An evaluator object that provides
                aggregate scores and per‑example failures.
            problem_description (str, optional): Task description passed
                as meta information.
            **kwargs: Additional HyPER hyperparameters:
                n_iterations (int), patience (int), n_candidates (int),
                top_n_candidates (int), k_samples (int),
                mini_batch_size (int).

        Returns:
            Tuple[str, list]: (best_prompt, iteration_history) where
                best_prompt is the optimized prompt string and
                iteration_history is a list of per‑iteration dictionaries.
        """
        # Extract HyPER hyperparameters from kwargs, with defaults
        n_iterations = kwargs.pop("n_iterations", 5)
        patience = kwargs.pop("patience", None)
        n_candidates = kwargs.pop("n_candidates", 3)
        top_n_candidates = kwargs.pop("top_n_candidates", 3)
        k_samples = kwargs.pop("k_samples", 3)
        mini_batch_size = kwargs.pop("mini_batch_size", 16)

        optimizer = HyPEROptimizer(
            model=model,
            evaluator=evaluator,
            n_iterations=n_iterations,
            patience=patience,
            n_candidates=n_candidates,
            top_n_candidates=top_n_candidates,
            k_samples=k_samples,
            mini_batch_size=mini_batch_size,
        )

        meta_info = None
        if problem_description:
            meta_info = {"problem_description": problem_description}

        final_prompt, _ = optimizer.optimize(
            prompt=initial_prompt,
            dataset_split=dataset_split,
            meta_info=meta_info,
        )
        return final_prompt

    def is_data_driven(self):
        """HyPER relies on evaluation data, so it is data‑driven.

        Returns:
            bool: True.
        """
        return True

    @property
    @override
    def name(self):
        """Name identifier of the method.

        Returns:
            str: "hyper".
        """
        return "hyper"

    @override
    def get_template(self, task):
        """Return the HyPE‑style prompt template (shared with HyPER).

        Args:
            task (Task): CLASSIFICATION or GENERATION.

        Returns:
            str: Template string.
        """
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE_HYPE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE_HYPE