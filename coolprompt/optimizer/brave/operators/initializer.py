from typing import List, Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    PROMPT_BY_DESCRIPTION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    reranking_population,
    PROMPT_TAGS
)
from coolprompt.optimizer.hype.hype import HyPEOptimizer
from coolprompt.utils.parsing import extract_answer


class PopulationInitializationOperator(Operator):
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

        hype = HyPEOptimizer(model)
        prompt_after_hype = hype.optimize(
            prompt=initial_prompt,
            meta_info={
                'problem_description': problem_description
            }
        )
        prompt_after_hype = Prompt(prompt_after_hype, origin=PromptOrigin.HYPE)

        initial_prompt = Prompt(initial_prompt, PromptOrigin.MANUAL)
        population = [initial_prompt, prompt_after_hype, prompt_by_pd]

        for prompt in population:
            evaluate_fn(prompt, "train")

        population = reranking_population(population)

        if self.logger is not None:
            self.logger.log_population(
                iteration=0,
                population=population
            )

        return population
