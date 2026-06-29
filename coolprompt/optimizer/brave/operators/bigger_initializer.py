from typing import List, Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.operators.creative_role_and_style import (
    CreativeRoleAndStyleMutationOperator
)
from coolprompt.optimizer.brave.operators.creative_zero_order import (
    CreativeZeroOrderMutationOperator
)
from coolprompt.optimizer.brave.operators.gradient_step import (
    GradientStepOperator
)
from coolprompt.optimizer.brave.operators.hype import HypeOperator
from coolprompt.optimizer.brave.prompt_templates import (
    PROMPT_BY_DESCRIPTION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    reranking_population,
    PROMPT_TAGS
)
from coolprompt.utils.parsing import extract_answer


class BiggerPopulationInitializationOperator(Operator):
    def run(
        self,
        initial_prompt: str,
        problem_description: str,
        population_size: int,  # just for interface
        model: BaseLanguageModel,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> List[Prompt]:
        prompt_by_description_template =\
            PROMPT_BY_DESCRIPTION_TEMPLATE.format(
                PROBLEM_DESCRIPTION=problem_description
            )
        prompt_by_pd = extract_answer(
            answer=llm_query_fn([prompt_by_description_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        prompt_by_pd = Prompt(prompt_by_pd, origin=PromptOrigin.BY_PD)

        initial_prompt = Prompt(initial_prompt, PromptOrigin.MANUAL)
        evaluate_fn(initial_prompt, "train")

        hype_operator = HypeOperator(model, logger=self.logger)
        hyped_prompt = hype_operator.run(
            iteration=-1,
            prompt=initial_prompt,
            problem_description=problem_description,
            evaluate_fn=evaluate_fn
        )

        gradient_step_operator = GradientStepOperator(self.logger)
        gradient_step_prompt = gradient_step_operator.run(
            iteration=-1,
            prompt=initial_prompt,
            problem_description=problem_description,
            llm_query_fn=llm_query_fn,
            evaluate_fn=evaluate_fn
        )

        creative_zero_order_operator = CreativeZeroOrderMutationOperator(
            logger=self.logger
        )
        creative_zero_order_prompt = creative_zero_order_operator.run(
            iteration=-1,
            prompt=initial_prompt,
            problem_description=problem_description,
            llm_query_fn=llm_query_fn,
            evaluate_fn=evaluate_fn
        )

        creative_ras_operator = CreativeRoleAndStyleMutationOperator(
            logger=self.logger
        )
        creative_roled_and_styled_prompt = creative_ras_operator.run(
            iteration=-1,
            prompt=initial_prompt,
            problem_description=problem_description,
            llm_query_fn=llm_query_fn,
            evaluate_fn=evaluate_fn
        )

        population = [
            prompt_by_pd,
            hyped_prompt,
            gradient_step_prompt,
            creative_zero_order_prompt,
            creative_roled_and_styled_prompt
        ]

        for prompt in population:
            evaluate_fn(prompt, "train")

        population.append(initial_prompt)

        population = reranking_population(population)

        if self.logger is not None:
            self.logger.log_population(
                iteration=0,
                population=population
            )

        return population
