from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.hype import hype_optimizer


class HyPEMethod(AutoPromptingMethod):
    name = "hype"
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