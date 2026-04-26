from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.distill_prompt import distillprompt


class DistillMethod(AutoPromptingMethod):
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description=None,
        **kwargs,
    ):
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