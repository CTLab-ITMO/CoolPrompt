from coolprompt.optimizer.autoprompting_method import (
    ConfiguredAutoPromptingMethod,
)
from coolprompt.optimizer.hype.hyper import HyPEROptimizer


class HyPERMethod(ConfiguredAutoPromptingMethod):
    """
    Interface for HyPER (HyPE with Refinement) method.

    HyPER uses iterative refinement via evaluation-based recommendations.
    It generates candidates, evaluates them, collects feedback on failed examples,
    and uses that feedback to improve the prompt through HyPE.

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
        config: (dict) provided configuration.
        dataset_split: dataset train/val split for optimization process.
        test_dataset: a dataset to use while testing the final prompt.
        test_target: string targets for testing dataset.
        evaluator: evaluator (Evaluator) to compute metrics.
    """

    def _run(self, start_prompt: str) -> str:
        """Runs HyPER optimization process.

        Args:
            start_prompt (str): initial prompt.

        Returns:
            str: optimized prompt.
        """
        meta_info = self.config.get('meta_info', {})
        if 'task_description' not in meta_info:
            meta_info['task_description'] = self.config.get('problem_description')

        method_config = self.config.get('method', {})
        n_iterations = method_config.get('n_iterations', 5)
        patience = method_config.get('patience', None)
        n_candidates = method_config.get('n_candidates', 3)
        top_n_candidates = method_config.get('top_n_candidates', 3)
        k_samples = method_config.get('k_samples', 3)
        mini_batch_size = method_config.get('mini_batch_size', 16)

        hyper_opt = HyPEROptimizer(
            model=self.model,
            evaluator=self.evaluator,
            n_iterations=n_iterations,
            patience=patience,
            n_candidates=n_candidates,
            top_n_candidates=top_n_candidates,
            k_samples=k_samples,
            mini_batch_size=mini_batch_size,
        )

        final_prompt, _ = hyper_opt.optimize(
            prompt=start_prompt,
            dataset_split=self.dataset_split,
            meta_info=meta_info,
        )

        return final_prompt
