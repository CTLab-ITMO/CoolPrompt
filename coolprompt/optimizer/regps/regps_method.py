from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.regps import regps


class ReGPSMethod(AutoPromptingMethod):
    name = "regps"
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ):
        return regps(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )