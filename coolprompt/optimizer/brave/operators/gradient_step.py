from typing import List, Callable

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin,
    BadExample
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    TEXTUAL_GRADIENT_TEMPLATE,
    GRADIENT_STEP_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS,
    FEEDBACK_TAGS
)
from coolprompt.utils.parsing import extract_answer


class GradientStepOperator(Operator):
    def _make_bad_examples(self, bad_examples: List[BadExample]) -> str:
        """Converts an array of bad examples into string format

        Args:
            bad_examples (List[BadExample]): list of bad examples.

        Returns:
            str: string representation of bad examples
        """

        return "\n\n".join([
            '\n'.join((
                f"Input: {example.input}",
                f"Model Output: {example.output}",
                f"Correct Output: {example.correct}"
            ))
            for example in bad_examples
        ])

    def _gen_textual_gradient(
        self,
        prompt: Prompt,
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
    ) -> str:
        """Generates textual gradient for provided prompt

        Args:
            prompt (Prompt): prompt to generate textual gradient for.

        Returns:
            str: textual gradient for given prompt
        """

        if prompt.gradient is not None:
            return prompt.gradient

        request = TEXTUAL_GRADIENT_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            PROMPT=prompt.text,
            EXAMPLES=self._make_bad_examples(prompt.bad_examples)
        )
        gradient = extract_answer(
            answer=llm_query_fn([request])[0],
            tags=FEEDBACK_TAGS,
            format_mismatch_label=""
        )
        prompt.gradient = gradient
        return gradient

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Prompt:
        gradient = self._gen_textual_gradient(
            prompt=prompt,
            problem_description=problem_description,
            llm_query_fn=llm_query_fn
        )

        gradient_step_template = GRADIENT_STEP_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            PROMPT=prompt.text,
            TEXTUAL_GRADIENT=gradient
        )
        gradiented = extract_answer(
            answer=llm_query_fn([gradient_step_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        gradiented = Prompt(
            gradiented,
            origin=PromptOrigin.GRADIENT_STEP
        )
        evaluate_fn(gradiented, "train")

        if self.logger is not None:
            self.logger.log_gradient_step(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=gradiented.text,
                mutated_score=gradiented.score,
                textual_gradient=gradient
            )

        return gradiented
