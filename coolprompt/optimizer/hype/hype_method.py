from typing import override

from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod
from coolprompt.optimizer.hype import HyPEOptimizer


class HyPEMethod(AutoPromptingMethod):
    """HyPE (Hypothetical Prompt Enhancer) method for auto‑prompting.

    This method uses the HyPE optimizer to refine an initial prompt without
    requiring a data‑driven training phase.
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
        """Run the HyPE prompt optimization.

        Args:
            model: The language model to be used by the optimizer.
            initial_prompt (str): The starting prompt text.
            dataset_split (optional): Not used by HyPE.
            evaluator (optional): Not used by HyPE.
            problem_description (str, optional): Task description; passed as
                meta information to the optimizer.
            **kwargs: Additional arguments, including hype_meta_info (dict).

        Returns:
            str: The optimized prompt string.
        """
        hype_meta_info = kwargs.pop("hype_meta_info", None)
        optimizer = HyPEOptimizer(model=model, **kwargs)
        meta_info = hype_meta_info.copy() if hype_meta_info else {}
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description
        return optimizer.optimize(
            prompt=initial_prompt,
            meta_info=meta_info if meta_info else None,
            n_prompts=1,
        )

    def is_data_driven(self):
        return False

    @property
    @override
    def name(self):
        return "hype"