from typing import List, Callable

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    PARAPHRASE_BY_DESCRIPTION_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS
)
from coolprompt.utils.parsing import extract_answer


class ParaphrasingByPDOperator(Operator):
    def run(
        self,
        iteration: int,
        prompt: Prompt,
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Prompt:
        paraphrasing_template = PARAPHRASE_BY_DESCRIPTION_TEMPLATE.format(
            PROBLEM_DESCRIPTION=problem_description,
            PROMPT=prompt.text
        )
        paraphrased = extract_answer(
            answer=llm_query_fn([paraphrasing_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )
        paraphrased = Prompt(
            paraphrased,
            origin=PromptOrigin.PARAPHRASED
        )
        evaluate_fn(paraphrased, "train")

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=paraphrased.text,
                mutated_score=paraphrased.score,
                file_name="paraphrases"
            )

        return paraphrased
