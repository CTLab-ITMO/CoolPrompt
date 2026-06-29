from typing import List, Tuple, Callable
import numpy as np

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.brave.prompt_templates import (
    FEW_SHOT_EXAMPLES_REMOVING_TEMPLATE,
    FEW_SHOT_EXAMPLES_INCORPORATING_TEMPLATE
)
from coolprompt.optimizer.brave.utils import (
    PROMPT_TAGS
)
from coolprompt.utils.parsing import extract_answer


class FewShotExamplesOperator(Operator):
    def __init__(
        self,
        max_few_shot_examples_num: int,
        data_sample: List[Tuple[str, str]],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_few_shot_examples_num = max_few_shot_examples_num
        self.examples = data_sample

    def _filter_possible_examples(
        self,
        prompt_few_shots: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        return [
            example
            for example in self.examples
            if example not in prompt_few_shots
        ]

    def _prepare_examples(self, examples: List[Tuple[str, str]]) -> str:
        return '\n\n'.join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None]
    ) -> Prompt:
        possible_examples = self._filter_possible_examples(
            prompt_few_shots=prompt.few_shot_examples
        )
        ind = np.random.choice(len(possible_examples))
        example_to_add = possible_examples[ind]

        original_few_shots = list(prompt.few_shot_examples)
        removed = ("", "")
        if len(prompt.few_shot_examples) == self.max_few_shot_examples_num:
            ind = np.random.choice(len(prompt.few_shot_examples))
            removed = prompt.few_shot_examples[ind]
            prompt.few_shot_examples[ind] = example_to_add
        else:
            prompt.add_few_shot_example(example_to_add)

        removing_template = FEW_SHOT_EXAMPLES_REMOVING_TEMPLATE.format(
            PROMPT=prompt.text
        )
        prompt_without_few_shots = extract_answer(
            answer=llm_query_fn([removing_template])[0],
            tags=PROMPT_TAGS,
            format_mismatch_label=""
        )

        few_shot_template = FEW_SHOT_EXAMPLES_INCORPORATING_TEMPLATE.format(
            PROMPT=prompt_without_few_shots,
            EXAMPLES=self._prepare_examples(prompt.few_shot_examples)
        )
        try:
            prompt_with_few_shots = extract_answer(
                answer=llm_query_fn([few_shot_template])[0],
                tags=PROMPT_TAGS,
                format_mismatch_label=""
            )
        except Exception:
            prompt_with_few_shots = None

        if prompt_with_few_shots:
            mutated_offspring = Prompt(
                prompt_with_few_shots,
                origin=PromptOrigin.FEW_SHOT
            )
            evaluate_fn(mutated_offspring, "train")
        else:
            prompt.few_shot_examples = original_few_shots
            mutated_offspring = Prompt(
                "FAILED TO PRODUCE",
                origin=PromptOrigin.FEW_SHOT
            )
            mutated_offspring.set_score(0)

        if self.logger is not None:
            self.logger.log_few_shot_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=mutated_offspring.text,
                mutated_score=mutated_offspring.score,
                added_few_shot=example_to_add,
                removed_few_shot=removed
            )

        return mutated_offspring
