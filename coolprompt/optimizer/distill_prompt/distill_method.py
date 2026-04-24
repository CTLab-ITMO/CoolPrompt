from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.distill_prompt import distillprompt


class DistillMethod(AutoPromptingMethod):
    name = "distill"
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