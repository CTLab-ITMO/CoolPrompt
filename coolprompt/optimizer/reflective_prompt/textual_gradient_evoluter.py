from typing import List, Optional, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    BadExample,
    PromptOrigin
)
from coolprompt.utils.parsing import extract_answer
from coolprompt.utils.prompt_templates.reflective_templates import (
    SHORT_TERM_TEXTGRAD_TEMPLATE,
    REFLECTIVEPROMPT_TEXTUAL_GRADIENT_TEMPLATE,
    MUTATION_TEXTGRAD_TEMPLATE
)


class ReflectiveEvoluterWithTextualGradient(ReflectiveEvoluter):
    FEEDBACK_TAGS = ("<feedback>", "</feedback>")

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        train_dataset: List[str],
        train_targets: List[str],
        validation_dataset: List[str],
        validation_targets: List[str],
        problem_description: str,
        initial_prompt: Optional[str] = None,
        population_size: int = 10,
        num_epochs: int = 10,
        output_path: str = "./reflectiveprompt_outputs",
        use_cache: bool = True,
        bad_examples_number: int = 5
    ) -> None:
        super().__init__(
            model,
            evaluator,
            train_dataset,
            train_targets,
            validation_dataset,
            validation_targets,
            problem_description,
            initial_prompt,
            population_size,
            num_epochs,
            output_path,
            use_cache
        )

        self.bad_examples_num = bad_examples_number

    def _evaluate(self, prompt: Prompt, split="train") -> None:
        """Evaluates given prompt on self.dataset and records the score.

        Args:
            prompt (Prompt): a prompt to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        if split == "train":
            dataset, targets = self.train_dataset, self.train_targets
        else:
            dataset, targets = self.validation_dataset, self.validation_targets
        score, bad_examples = self.evaluator.evaluate(
            prompt=prompt.text,
            dataset=dataset,
            targets=targets,
            failed_examples=self.bad_examples_num
        )
        prompt.set_score(score)
        prompt.set_bad_examples(bad_examples)

    def _make_bad_examples(self, bad_examples: List[BadExample]) -> str:
        return "\n\n".join([
            '\n'.join((
                f"Input: {example.input}",
                f"Model Output: {example.output}",
                f"Correct Output: {example.correct}"
            ))
            for example in bad_examples
        ])

    def _gen_textual_gradient(self, prompt: Prompt) -> str:
        request = REFLECTIVEPROMPT_TEXTUAL_GRADIENT_TEMPLATE.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            PROMPT=prompt.text,
            EXAMPLES=self._make_bad_examples(prompt.bad_examples)
        )
        return extract_answer(
            self._llm_query([request])[0],
            self.FEEDBACK_TAGS,
            format_mismatch_label=""
        )

    def _gen_short_term_reflection_prompt(
        self, ind1: Prompt, ind2: Prompt
    ) -> Tuple[str, str, str, str, str]:
        """Generates short-term reflection request into model.

        Args:
            ind1 (Prompt): first individual.
            ind2 (Prompt): second individual.

        Returns:
            Tuple[str, str, str]:
                string request, worse prompt text, better prompt text.
        """
        if ind1.score > ind2.score:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        better_feedback = self._gen_textual_gradient(better_ind)
        worse_feedback = self._gen_textual_gradient(worse_ind)

        request = SHORT_TERM_TEXTGRAD_TEMPLATE.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            WORSE_PROMPT=worse_ind.text,
            WORSE_PROMPT_FEEDBACK=worse_feedback,
            BETTER_PROMPT=better_ind.text,
            BETTER_PROMPT_FEEDBACK=better_feedback
        )

        return (
            request,
            worse_ind.text,
            better_ind.text,
            worse_feedback,
            better_feedback
        )

    def _short_term_reflection(
        self,
        population: list[Prompt],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Short-term reflection before crossovering two individuals.

        Args:
            population (list[Prompt]): parenting population.

        Returns:
            Tuple[List[str], List[str], List[str]]:
                generated short-term hints,
                worse promtp texts,
                better prompt texts.
        """
        requests = []
        worse_prompts = []
        better_prompts = []
        feedbacks = []
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]

            (
                request,
                worse_prompt,
                better_prompt,
                worse_feedback,
                better_feedback
            ) = (
                self._gen_short_term_reflection_prompt(parent_1, parent_2)
            )
            requests.append(request)
            worse_prompts.append(worse_prompt)
            better_prompts.append(better_prompt)
            feedbacks.append((worse_feedback, better_feedback))

        self._cache_data(
            feedbacks,
            self._make_output_path("feedbacks"),
        )
        responses = self._llm_query(requests)
        responses = [
            extract_answer(response, self.HINT_TAGS, format_mismatch_label="")
            for response in responses
        ]
        return responses, worse_prompts, better_prompts

    def _mutate(self) -> List[Prompt]:
        """Elitist-based mutation.

        Returns:
            List[Prompt]: generated population.
        """
        feedback = self._gen_textual_gradient(self.elitist)
        self._cache_data(
            feedback,
            self._make_output_path("elitist_feedback"),
        )
        request = MUTATION_TEXTGRAD_TEMPLATE.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            LONG_TERM_REFLECTION=self._long_term_reflection_str,
            ELITIST_PROMPT=self.elitist.text,
            FEEDBACK=feedback
        )
        responses = self._llm_query([request] * self.population_size)
        responses = [
            extract_answer(
                response, self.PROMPT_TAGS, format_mismatch_label=""
            )
            for response in responses
        ]
        population = [
            Prompt(response, origin=PromptOrigin.MUTATED)
            for response in responses
        ]
        return population
