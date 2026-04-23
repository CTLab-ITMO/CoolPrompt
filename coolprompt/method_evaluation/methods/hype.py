from coolprompt.method_evaluation.methods.autoprompting_method import (
    AutoPromptingMethod
)
from coolprompt.optimizer.hype.hype import HyPEOptimizer


class HyPEMethod(AutoPromptingMethod):
    """
    Interface for HyPE (Hypothetical Prompt Enhancer) method.

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
        config: (dict) provided configuration.
        dataset_split: dataset train/val split for optimization process.
        test_dataset: a dataset to use while testing the final prompt.
        test_target: string targets for testing dataset.
        evaluator: evaluator (Evaluator) to compute metrics.
    """

    def _run(self, start_prompt: str) -> str:
        """Runs HyPE optimization process.

        Args:
            start_prompt (str): initial prompt.

        Returns:
            str: optimized prompt.
        """
        meta_info = self.config.get('meta_info', {})

        hype_opt = HyPEOptimizer(model=self.model)
        final_prompt = hype_opt.optimize(
            prompt=start_prompt,
            meta_info=meta_info,
        )

        return final_prompt
