from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.enums import Task


class HyPEMethod(AutoPromptingMethod):
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        return hype_optimizer(
            model=model,
            prompt=initial_prompt,
            problem_description=problem_description,
        )

    def is_data_driven(self):
        
        return False

    @property
    @override
    def name(self):
        return "hype"

    @override
    def get_template(self, task):
        match task:
            case Task.CLASSIFICATION:
                return CLASSIFICATION_TASK_TEMPLATE_HYPE
            case Task.GENERATION:
                return GENERATION_TASK_TEMPLATE_HYPE