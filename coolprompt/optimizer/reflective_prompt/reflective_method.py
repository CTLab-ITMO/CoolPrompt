from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.reflective_prompt import reflectiveprompt


class ReflectiveMethod(AutoPromptingMethod):
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ):
        return reflectiveprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "reflective"