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
from coolprompt.utils.parsing import extract_answer


class ParaphraseInitializationOperator(Operator):
    def run(
        self,
        initial_prompt: str,
        population_size: int,
        problem_description: str,
        model: BaseLanguageModel,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> List[Prompt]:
        prompt_by_description_template =\
            PROMPT_BY_DESCRIPTION_TEMPLATE.format(
                PROBLEM_DESCRIPTION=problem_description
            )
        answers = llm_query_fn(
            [prompt_by_description_template] * (population_size - 1)
        )
        prompts = [
            extract_answer(
                answer=ans,
                tags=PROMPT_TAGS,
                format_mismatch_label=""
            )
            for ans in answers
        ]
        prompts = [
            Prompt(prompt, origin=PromptOrigin.BY_PD) for prompt in prompts
        ]

        initial_prompt = Prompt(initial_prompt, PromptOrigin.MANUAL)
        prompts.append(initial_prompt)

        for prompt in prompts:
            evaluate_fn(prompt, "train")

        population = reranking_population(prompts)

        if self.logger is not None:
            self.logger.log_population(
                iteration=0,
                population=population
            )

        return population
