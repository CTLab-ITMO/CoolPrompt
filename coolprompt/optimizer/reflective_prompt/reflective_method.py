from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.reflective_prompt import reflectiveprompt


class ReflectiveMethod(AutoPromptingMethod):
    name = "reflective"
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