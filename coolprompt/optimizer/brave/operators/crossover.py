from typing import List, Tuple, Callable
import numpy as np

from coolprompt.optimizer.reflective_prompt.prompt import (
    BadExample,
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    TEXTUAL_GRADIENT_TEMPLATE,
    SHORT_TERM_REFLECTION_TEMPLATE,
    CROSSOVER_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS,
    HINT_TAGS,
    FEEDBACK_TAGS
)
from coolprompt.utils.parsing import extract_answer


class CrossoverOperator(Operator):
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
        population: List[Prompt],
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Tuple[Prompt, str]:
        scores = np.array([prompt.score for prompt in population])
        probas = (scores + 1e-5) / np.sum(scores + 1e-5)
        parents = np.random.choice(
            population, size=2, replace=False, p=probas
        )

        parents = [
            (
                parent,
                self._gen_textual_gradient(
                    parent,
                    problem_description,
                    llm_query_fn=llm_query_fn
                )
            )
            for parent in parents
        ]

        short_term_template = SHORT_TERM_REFLECTION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            PROMPT1=parents[0][0].text,
            FEEDBACK1=parents[0][1],
            PROMPT2=parents[1][0].text,
            FEEDBACK2=parents[1][1],
        )
        short_term_reflection = extract_answer(
            answer=llm_query_fn([short_term_template])[0],
            tags=HINT_TAGS,
            format_mismatch_label=""
        )

        crossover_template = CROSSOVER_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            PARENT1=parents[0][0].text,
            PARENT2=parents[1][0].text,
            SHORT_TERM_REFLECTION=short_term_reflection
        )
        offspring = extract_answer(
            answer=llm_query_fn([crossover_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        offspring = Prompt(offspring, origin=PromptOrigin.CROSSOVER)
        evaluate_fn(offspring, "train")

        if self.logger is not None:
            self.logger.log_crossover(
                iteration=iteration,
                parent1_prompt=parents[0][0].text,
                parent1_score=parents[0][0].score,
                parent2_prompt=parents[1][0].text,
                parent2_score=parents[1][0].score,
                parent1_textual_gradient=parents[0][1],
                parent2_textual_gradient=parents[1][1],
                offspring_prompt=offspring.text,
                offspring_score=offspring.score,
                short_term_reflection=short_term_reflection
            )

        return offspring, short_term_reflection
