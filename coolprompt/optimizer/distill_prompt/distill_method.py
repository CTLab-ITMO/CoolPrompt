from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.distill_prompt import distillprompt


class DistillMethod(AutoPromptingMethod):
    """Distillation‑based method for auto‑prompting.

    This method optimizes a prompt by distilling knowledge from a larger
    model or from feedback on a training dataset. It relies on a labeled
    data split and an evaluator to guide prompt improvement.
    """

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description=None,
        **kwargs,
    ):
        """Run the distillation prompt optimization.

        Args:
            model: The language model to be optimized (e.g., a HuggingFace
                model or similar).
            initial_prompt (str): The starting prompt text.
            dataset_split: A labeled dataset split (e.g., train/validation)
                used by the distillation process.
            evaluator: An evaluator object that provides scoring or feedback
                on generated prompts.
            problem_description (str, optional): Natural language description
                of the task. Defaults to None.
            **kwargs: Additional keyword arguments passed to `distillprompt`.

        Returns:
            The optimized prompt result from `distillprompt`.
        """
        return distillprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    def is_data_driven(self):
        return True

    @property
    @override
    def name(self):
        return "distill"
