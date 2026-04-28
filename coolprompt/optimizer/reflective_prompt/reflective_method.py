from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.reflective_prompt import reflectiveprompt


class ReflectiveMethod(AutoPromptingMethod):
    """Reflective prompting method that iteratively improves prompts.

    This method uses a reflective process: it generates candidate prompts,
    evaluates them on a dataset, and reflects on the results to refine
    the prompt further. It requires a labeled dataset split and an evaluator.
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
        """Run the reflective prompt optimization.

        Args:
            model: The language model used for generating and reflecting
                on prompts.
            initial_prompt (str): The starting prompt text.
            dataset_split: A labeled dataset split (e.g., train/validation)
                used to evaluate prompt performance.
            evaluator: An evaluator object that computes a score or
                feedback for a given prompt on the dataset.
            problem_description (str): A natural language description of
                the task to guide the reflection process.
            **kwargs: Additional keyword arguments passed to
                `reflectiveprompt`.

        Returns:
            The optimized prompt result from `reflectiveprompt`.
        """
        return reflectiveprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    def is_data_driven(self) -> bool:
        """Indicate whether this method requires data for optimization.

        Returns:
            bool: True because reflective prompting needs a labeled dataset.
        """
        return True

    @property
    @override
    def name(self) -> str:
        """Name identifier of the method.

        Returns:
            str: The string "reflective".
        """
        return "reflective"
