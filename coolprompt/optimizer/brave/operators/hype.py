from typing import Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.hype.hype import HyPEOptimizer


class HypeOperator(Operator):
    def __init__(self, model: BaseLanguageModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hype = HyPEOptimizer(model)

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        hyped = self.hype.optimize(
            prompt.text,
            meta_info={
                'problem_description': problem_description
            }
        )
        hyped = Prompt(hyped, origin=PromptOrigin.HYPE)
        evaluate_fn(hyped, "train")

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=hyped.text,
                mutated_score=hyped.score,
                file_name="hype"
            )

        return hyped
