from typing import List, Tuple, Callable

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    CREATIVE_STYLE_AND_ROLE_TEMPLATE,
    CREATIVE_ZERO_ORDER_MUTATION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS,
    STYLE_TAGS,
    ROLE_TAGS
)
from coolprompt.utils.parsing import extract_answer


class CreativeRoleAndStyleMutationOperator(Operator):
    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Tuple[Prompt, str]:
        style_and_role_template = CREATIVE_STYLE_AND_ROLE_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description
        )
        print(style_and_role_template)
        model_answer = llm_query_fn([style_and_role_template])[0]
        print(model_answer)
        style = extract_answer(
            answer=model_answer,
            tags=STYLE_TAGS,
            format_mismatch_label=""
        )
        role = extract_answer(
            answer=model_answer,
            tags=ROLE_TAGS,
            format_mismatch_label=""
        )

        mutation_template = CREATIVE_ZERO_ORDER_MUTATION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            STYLE=style,
            ROLE=role,
            PROMPT=prompt.text
        )
        mutated_offspring = extract_answer(
            answer=llm_query_fn([mutation_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        mutated_offspring = Prompt(
            mutated_offspring,
            origin=PromptOrigin.CREATIVE_IN_STYLE_OF
        )
        evaluate_fn(mutated_offspring, "train")

        if self.logger is not None:
            self.logger.log_creative_role_style_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=mutated_offspring.text,
                mutated_score=mutated_offspring.score,
                style=style,
                role=role
            )

        return mutated_offspring
