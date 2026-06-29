from typing import List, Tuple, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coolprompt.optimizer.reflective_prompt.prompt import (
    BadExample,
    Prompt,
    PromptOrigin,
)
from coolprompt.optimizer.brave.operators.few_shot_examples import (
    FewShotExamplesOperator,
)
from coolprompt.optimizer.brave.prompt_templates import (
    FEW_SHOT_EXAMPLES_REMOVING_TEMPLATE,
    FEW_SHOT_EXAMPLES_INCORPORATING_TEMPLATE,
)
from coolprompt.optimizer.brave.utils import PROMPT_TAGS
from coolprompt.utils.parsing import extract_answer


class HardFewShotExamplesOperator(FewShotExamplesOperator):
    """Few-shot operator that selects examples closest to the prompt's
    current failure cases instead of sampling uniformly at random.

    When no bad_examples are available (e.g. early in optimization),
    falls back to uniform random selection from the parent class.
    """

    def _select_example(
        self,
        possible_examples: List[Tuple[str, str]],
        bad_examples: List[BadExample],
    ) -> Tuple[str, str]:
        """Pick the candidate whose input is most similar to bad_examples.

        Similarity is TF-IDF cosine, averaged over all bad_example inputs.
        Falls back to random if bad_examples is empty or TF-IDF fails.
        """
        if not bad_examples or len(possible_examples) == 1:
            return possible_examples[np.random.choice(len(possible_examples))]

        candidate_inputs = [ex[0] for ex in possible_examples]
        bad_inputs = [be.input for be in bad_examples]

        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1,
            )
            all_texts = candidate_inputs + bad_inputs
            tfidf = vectorizer.fit_transform(all_texts)
            n = len(candidate_inputs)
            sim = cosine_similarity(tfidf[:n], tfidf[n:])  # (n_cands, n_bad)
            scores = sim.mean(axis=1)
            best_idx = int(np.argmax(scores))
        except Exception:
            best_idx = np.random.choice(len(possible_examples))

        return possible_examples[best_idx]

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        possible_examples = self._filter_possible_examples(
            prompt_few_shots=prompt.few_shot_examples
        )
        example_to_add = self._select_example(possible_examples, prompt.bad_examples)

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
        answer_after_removal = llm_query_fn([removing_template])[0]
        prompt_without_few_shots = extract_answer(
            answer=answer_after_removal,
            tags=PROMPT_TAGS,
            format_mismatch_label="",
        )

        few_shot_template = FEW_SHOT_EXAMPLES_INCORPORATING_TEMPLATE.format(
            PROMPT=prompt_without_few_shots,
            EXAMPLES=self._prepare_examples(prompt.few_shot_examples),
        )
        try:
            answer_with_few_shot = llm_query_fn([few_shot_template])[0]
            prompt_with_few_shots = extract_answer(
                answer=answer_with_few_shot,
                tags=PROMPT_TAGS,
                format_mismatch_label="",
            )
        except Exception:
            prompt_with_few_shots = None

        if prompt_with_few_shots:
            mutated_offspring = Prompt(
                prompt_with_few_shots,
                origin=PromptOrigin.FEW_SHOT,
            )
            evaluate_fn(mutated_offspring, "train")
        else:
            prompt.few_shot_examples = original_few_shots
            mutated_offspring = Prompt(
                "FAILED TO PRODUCE",
                origin=PromptOrigin.FEW_SHOT,
            )
            mutated_offspring.set_score(0)

            if self.logger is not None:
                self.logger.log_few_shot_mutation(
                    iteration=iteration,
                    prompt=prompt.text,
                    prev_score=prompt.score,
                    mutated_prompt=mutated_offspring.text,
                    mutated_score=mutated_offspring.score,
                    added_few_shot=answer_after_removal,
                    removed_few_shot=answer_with_few_shot,
                    file_name="failed_few_shot_mutations",
                )

        if self.logger is not None:
            self.logger.log_few_shot_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=mutated_offspring.text,
                mutated_score=mutated_offspring.score,
                added_few_shot=example_to_add,
                removed_few_shot=removed,
                file_name="hard_few_shot_mutations",
            )

        return mutated_offspring
