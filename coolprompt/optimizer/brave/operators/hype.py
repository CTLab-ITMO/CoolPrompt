from typing import Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.hyper.meta_prompt import MetaPromptOptimizer


class HypeOperator(Operator):
    """Apply the HYPE prompt optimization strategy."""

    def __init__(self, model: BaseLanguageModel, **kwargs) -> None:
        """Create a HYPE optimizer backed by the supplied model.

        Args:
            model (BaseLanguageModel): model used by HYPE.
            **kwargs (Any): base-operator arguments such as ``logger``.
        """

        super().__init__(**kwargs)
        self.hype = MetaPromptOptimizer(model)

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        """Optimize and evaluate a prompt with HYPE.

        Args:
            iteration (int): optimization iteration used for logging.
            prompt (Prompt): prompt to optimize.
            problem_description (str): description of the target task.
            evaluate_fn (Callable[[Prompt, str], None]): prompt evaluator.

        Returns:
            Prompt: evaluated HYPE mutation.
        """

        hyped = self.hype.optimize(
            prompt.text, meta_info={"problem_description": problem_description}
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
                file_name="hype",
            )

        return hyped
