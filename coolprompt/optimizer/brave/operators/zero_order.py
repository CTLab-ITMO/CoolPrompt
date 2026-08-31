from typing import List, Callable

from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import PROMPT_BY_DESCRIPTION_TEMPLATE
from coolprompt.optimizer.brave.utils import PROMPT_TAGS
from coolprompt.utils.parsing import extract_answer


class ZeroOrderMutationOperator(Operator):
    """Mutate a prompt without using evaluation-derived gradients."""

    def run(
        self,
        iteration: int,
        prompt: Prompt,  # won't be used, but needed for the interface
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        """Generate and evaluate a zero-order prompt mutation.

        Args:
            iteration (int): optimization iteration used for logging.
            prompt (Prompt): interface-compatible source prompt; not read.
            problem_description (str): description of the target task.
            llm_query_fn (Callable[[List[str]], List[str]]): batched LLM
                callback.
            evaluate_fn (Callable[[Prompt, str], None]): prompt evaluator.

        Returns:
            Prompt: evaluated generated prompt.
        """

        generating_template = PROMPT_BY_DESCRIPTION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
        )
        generated = extract_answer(
            answer=llm_query_fn([generating_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label="",
        )
        generated = Prompt(generated, origin=PromptOrigin.BY_PD)
        evaluate_fn(generated, "train")

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt="",
                prev_score=-1.0,
                mutated_prompt=generated.text,
                mutated_score=generated.score,
                file_name="zero_orders",
            )

        return generated
