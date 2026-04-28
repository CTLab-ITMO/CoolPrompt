from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.regps import regps


class ReGPSMethod(AutoPromptingMethod):
    """ReGPS (Recursive Guided Prompt Search) method for auto‑prompting.

    This method uses a recursive guided search to iteratively refine
    prompts based on evaluation feedback from a labeled dataset.
    """

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ):
        """Run the ReGPS prompt optimization.

        Args:
            model: The language model used for prompt generation and search.
            initial_prompt (str): The starting prompt text.
            dataset_split: A labeled dataset split (e.g., train/validation)
                used to evaluate candidate prompts.
            evaluator: An evaluator object that scores prompts against
                the dataset.
            problem_description (str): Natural language description of the
                task to guide the search.
            **kwargs: Additional keyword arguments passed to `regps`.

        Returns:
            The optimized prompt result from `regps`.
        """
        return regps(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    def is_data_driven(self):
        """Indicate whether this method requires data for optimization.

        Returns:
            bool: True because ReGPS needs a labeled dataset split.
        """
        return True

    @property
    @override
    def name(self):
        """Name identifier of the method.

        Returns:
            str: The string "regps".
        """
        return "regps"