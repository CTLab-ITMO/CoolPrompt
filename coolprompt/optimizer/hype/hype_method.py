from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.enums import Task


class HyPEMethod(AutoPromptingMethod):
    """HyPE (Hypothesis‑Prompt Evolution) method for auto‑prompting.

    This method uses the HyPE optimizer to evolve an initial prompt without
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
            model: The language model to be optimized (e.g., a HuggingFace
                model or similar).
            initial_prompt (str): The starting prompt text.
            dataset_split (optional): Data split used by data‑driven methods;
                not used by HyPE. Defaults to None.
            evaluator (optional): Evaluator object for scoring prompts;
                not used by HyPE. Defaults to None.
            problem_description (str, optional): Natural language description
                of the task to guide the evolution. Defaults to None.
            **kwargs: Additional arguments passed through to `hype_optimizer`.

        Returns:
            The optimized prompt result from `hype_optimizer`.
        """
        return hype_optimizer(
            model=model,
            prompt=initial_prompt,
            problem_description=problem_description,
        )

    def is_data_driven(self):
        """Indicate whether this method requires data for optimization.

        Returns:
            bool: False because HyPE is a data‑free method.
        """
        return False

    @property
    @override
    def name(self):
        """Name identifier of the method.

        Returns:
            str: The string "hype".
        """
        return "hype"

    @override
    def get_template(self, task):
        """Return the HyPE‑specific prompt template for a given task type.

        Args:
            task (Task): The task enum value (e.g., CLASSIFICATION, GENERATION).

        Returns:
            str: The corresponding template string.
        """
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE_HYPE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE_HYPE