from typing import List, Tuple, Callable

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    ELITIST_MUTATION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS
)
from coolprompt.utils.parsing import extract_answer


class LongTermMutationOperator(Operator):
    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        long_term_reflection: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Tuple[Prompt, str]:
        mutation_template = ELITIST_MUTATION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            ELITIST_PROMPT=prompt.text,
            LONG_TERM_REFLECTION=long_term_reflection
        )
        mutated_offspring = extract_answer(
            answer=llm_query_fn([mutation_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        mutated_offspring = Prompt(
            mutated_offspring,
            origin=PromptOrigin.LONG_TERM_MUTATION
        )
        evaluate_fn(mutated_offspring, "train")

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=mutated_offspring.text,
                mutated_score=mutated_offspring.score,
                file_name="long_term_mutations"
            )

        return mutated_offspring
