from typing import List, Tuple, Callable

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    LONG_TERM_REFLECTION_TEMPLATE,
    LONG_TERM_REFLECTION_UPDATE_TEMPLATE,
    ELITIST_MUTATION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS,
    HINT_TAGS
)
from coolprompt.utils.parsing import extract_answer


class ElitistMutationOperator(Operator):
    def run(
        self,
        iteration: int,
        elitist: Prompt,
        problem_description: str,
        long_term_reflection: str,
        short_term_reflections: List[str],
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Tuple[Prompt, str]:
        if long_term_reflection == "":
            long_term_template = LONG_TERM_REFLECTION_TEMPLATE.format(
                SHORT_TERM_REFLECTIONS='/n'.join(short_term_reflections)
            )
        else:
            long_term_template = LONG_TERM_REFLECTION_UPDATE_TEMPLATE.format(
                SHORT_TERM_REFLECTIONS='/n'.join(short_term_reflections),
                LONG_TERM_REFLECTION=long_term_reflection
            )
        new_long_term_reflection = extract_answer(
            answer=llm_query_fn([long_term_template])[0],
            tags=HINT_TAGS,
            format_mismatch_label=""
        )

        mutation_template = ELITIST_MUTATION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            ELITIST_PROMPT=elitist.text,
            LONG_TERM_REFLECTION=new_long_term_reflection
        )
        mutated_offspring = extract_answer(
            answer=llm_query_fn([mutation_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        mutated_offspring = Prompt(
            mutated_offspring,
            origin=PromptOrigin.ELITIST_MUTATION
        )
        evaluate_fn(mutated_offspring, "train")

        if self.logger is not None:
            self.logger.log_elitist_mutation(
                iteration=iteration,
                elitist_prompt=elitist.text,
                prev_score=elitist.score,
                mutated_prompt=mutated_offspring.text,
                mutated_score=mutated_offspring.score,
                new_long_term_reflection=new_long_term_reflection,
                short_term_reflections=short_term_reflections
            )

        return mutated_offspring, new_long_term_reflection
