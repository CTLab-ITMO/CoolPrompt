from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.hype.hyper import HyPEROptimizer


class HyPERMethod(AutoPromptingMethod):
    """HyPER (Hypothetical Prompt Enhancer with Refinement) method.

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

        hype_meta_info = kwargs.pop("hype_meta_info", None)
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

        meta_info = hype_meta_info.copy() if hype_meta_info else {}
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description

        final_prompt, _ = optimizer.optimize(
            prompt=initial_prompt,
            dataset_split=dataset_split,
            meta_info=meta_info if meta_info else None,
        )
        return final_prompt

    def is_data_driven(self):
        return True

    @property
    @override
    def name(self):
        return "hyper"