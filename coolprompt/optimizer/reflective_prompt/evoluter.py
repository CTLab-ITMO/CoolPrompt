import os
import json
import yaml
from typing import List, Tuple, Any

import numpy as np
import statistics
from scipy.special import softmax
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.reflective_templates import (
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE,
)
from coolprompt.utils.parsing import extract_answer


class ReflectiveEvoluter:
    """
    ReflectiveEvoluter class that represents evoluter for ReflectivePrompt

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
        evaluator: evaluator (Evaluator) to compute metrics.
        train_dataset: a dataset to use while training.
        train_targets: string targets for train dataset.
        validation_dataset: a dataset to use while validating final prompts.
        validation_targets: string targets for validation dataset.
        problem_description: a string that contains
            short description of problem to optimize.
        initial_prompt: initial prompt to start evolution from.
            Will be automatically generated if not provided.
            Defaults to None.
        population_size: an integer fixed size of prompt population.
            Defaults to 10.
        num_epochs: an integer number of epochs to evaluate.
            Defaults to 10.
        use_cache: a boolean variable.
            Either to use caching files or not.
        output_path: a path to store logs of evolution.
        elitist: a prompt with highest score in population.
        best_score_overall: best evaluation score during evolution.
        best_prompt_overall: text of prompt with best score overall.
        iteration: current iteration (epoch) of evolution.
        PROMPT_TAGS: start and end tags for prompt extraction.
        HINT_TAGS: start and end tags for hint extraction.
    """

    PROMPT_TAGS = ("<prompt>", "</prompt>")
    HINT_TAGS = ("<hint>", "</hint>")

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        train_dataset: List[str],
        train_targets: List[str],
        validation_dataset: List[str],
        validation_targets: List[str],
        problem_description: str,
        initial_prompt: str = None,
        population_size: int = 10,
        num_epochs: int = 10,
        output_path: str = "./reflectiveprompt_outputs",
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets
        self.use_cache = use_cache
        self.population_size = population_size
        self.num_epochs = num_epochs
        self.problem_description = problem_description
        self.output_path = output_path
        self.initial_prompt = initial_prompt

        self.elitist = None
        self._long_term_reflection_str = ""
        self.best_score_overall = None
        self.best_prompt_overall = None
        self.iteration = 0

        self._paraphrasing_template = REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE
        self._crossover_template = REFLECTIVEPROMPT_CROSSOVER_TEMPLATE
        self._mutation_template = REFLECTIVEPROMPT_MUTATION_TEMPLATE
        self._short_term_template = (
            REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE
        )
        self._long_term_template = (
            REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE
        )
        self._initial_prompt_template = (
            REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE
        )

    def _reranking(self, population: List[Prompt]) -> List[Prompt]:
        """
        Sorts given population of prompts by their scores in descending order.

        Args:
            population (List[Prompt]): population to sort.

        Returns:
            List[Prompt]: sorted population.
        """
        return list(
            sorted(population, key=lambda prompt: prompt.score, reverse=True)
        )

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
        score = self.evaluator.evaluate(
            prompt=prompt.text,
            dataset=dataset,
            targets=targets,
        )
        prompt.set_score(score)

    def _evaluation(
        self, population: List[Prompt], split: str = "train"
    ) -> None:
        """Evaluation operation for prompts population.
        Evaluates every prompt in population and records the results.

        Args:
            population (List[Prompt]): population of prompts to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        logger.info("Evaluating population...")
        for prompt in population:
            self._evaluate(prompt, split=split)

    def _create_initial_prompt(self) -> str:
        """Creates an initial prompt according to provided problem description

        Returns:
            str: initial prompt
        """
        request = self._initial_prompt_template.format(
            PROBLEM_DESCRIPTION=self.problem_description
        )
        answer = self._llm_query([request])[0]
        return extract_answer(
            answer, self.PROMPT_TAGS, format_mismatch_label=""
        )

    def _init_pop(self) -> List[Prompt]:
        """Creates initial population of prompts.

        Returns:
            List[Prompt]: initial population.
        """

        logger.info("Initializing population...")
        if self.initial_prompt is None:
            self.initial_prompt = self._create_initial_prompt()
        request = self._paraphrasing_template.format(
            PROMPT=self.initial_prompt, NUM_PROMPTS=self.population_size
        )
        answer = self._llm_query([request])[0]
        prompts = json.loads(answer)["prompts"]
        initial_population = [
            Prompt(prompt, origin=PromptOrigin.APE) for prompt in prompts
        ]
        self._evaluation(initial_population)
        initial_population = self._reranking(initial_population)
        return initial_population

    def _cache_data(self, data: Any, savepath: os.PathLike) -> None:
        """Writes the data to the yaml file.

        Args:
            data (Any): data to be cached.
            savepath (os.PathLike): a path to saving file.
        """
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w") as f:
            yaml.dump(data, f)

    def _cache_population(
        self, population: List[Prompt], savepath: os.PathLike
    ) -> None:
        """Caching a population of prompts to file.
        If self.use_cache is False this function will do nothing.

        Args:
            population (List[Prompt]): prompt population.
            savepath (os.PathLike): a path to saving file.
        """
        if self.use_cache is False:
            return

        best_score = population[0].score
        average_score = statistics.mean(
            [prompt.score for prompt in population]
        )
        data = {
            "best_score": best_score,
            "average_score": average_score,
            "prompts": [prompt.to_dict() for prompt in population],
        }
        self._cache_data(data, savepath)

    def _selection(self, population: List[Prompt]) -> List[Prompt]:
        """Provides selection operation.
        In current implementation we want to select parents
        with different scores.
        But when there is difficult to do so (trial number check),
        it will just sample anyways.

        Probabilities - normalized scores.

        Args:
            population (List[Prompt]): prompt population to select from.

        Returns:
            List[Prompt]: selected prompts.
        """
        selected_population = []

        scores = np.array([prompt.score for prompt in population])
        probas = scores / np.sum(scores)

        trial = 0
        anyways = False
        while len(selected_population) < 2 * self.population_size:
            parents = np.random.choice(
                population, size=2, replace=False, p=probas
            )
            if parents[0].score != parents[1].score or anyways:
                selected_population.extend(parents)
            trial += 1
            if trial > 1000:
                anyways = True

        return selected_population

    def _survive(
        self, population: List[Prompt], temperature: float = None
    ) -> List[Prompt]:
        """Final selection before going into new epoch.
        Probabilities are based on softmax function with temperature (if set).

        Args:
            population (List[Prompt]): population to select from.
            temperature (float, optional): temperature parameter for softmax.
                Defaults to None.

        Returns:
            List[Prompt]: selected (survived) prompts.
        """
        scores = np.array([prompt.score for prompt in population])
        if temperature is not None:
            scores /= temperature
        probas = softmax(scores)
        return np.random.choice(
            population, size=self.population_size, replace=False, p=probas
        )

    def _gen_short_term_reflection_prompt(
        self, ind1: Prompt, ind2: Prompt
    ) -> Tuple[str, str, str]:
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

        request = self._short_term_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            WORSE_PROMPT=worse_ind.text,
            BETTER_PROMPT=better_ind.text,
        )

        return request, worse_ind.text, better_ind.text

    def _make_output_path(self, filename: str) -> os.PathLike:
        """Creates full path for logging based on current iteration.

        Args:
            filename (str): the file name to save.

        Returns:
            os.PathLike: final path to save.
        """
        return os.path.join(
            self.output_path, f"Iteration{self.iteration}", f"{filename}.yaml"
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
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]

            (request, worse_prompt, better_prompt) = (
                self._gen_short_term_reflection_prompt(parent_1, parent_2)
            )
            requests.append(request)
            worse_prompts.append(worse_prompt)
            better_prompts.append(better_prompt)

        responses = self._llm_query(requests)
        responses = [
            extract_answer(response, self.HINT_TAGS, format_mismatch_label="")
            for response in responses
        ]
        return responses, worse_prompts, better_prompts

    def _crossover(
        self,
        short_term_reflection_tuple: Tuple[List[str], List[str], List[str]],
    ) -> List[Prompt]:
        """Provides crossover operation.

        Args:
            short_term_reflection_tuple
                (Tuple[List[str], List[str], List[str]]):
                    outputs of short-term reflection.

        Returns:
            List[Prompt]: new crossed prompts population.
        """
        (reflection_contents, worse_prompts, better_prompts) = (
            short_term_reflection_tuple
        )
        requests = []
        for reflection, worse_prompt, better_prompt in zip(
            reflection_contents, worse_prompts, better_prompts
        ):
            request = self._crossover_template.format(
                PROBLEM_DESCRIPTION=self.problem_description,
                WORSE_PROMPT=worse_prompt,
                BETTER_PROMPT=better_prompt,
                SHORT_TERM_REFLECTION=reflection,
            )
            requests.append(request)

        responses = self._llm_query(requests)
        responses = [
            extract_answer(
                response, self.PROMPT_TAGS, format_mismatch_label=""
            )
            for response in responses
        ]
        crossed_population = [Prompt(response) for response in responses]

        assert len(crossed_population) == self.population_size
        return crossed_population

    def _update_elitist(self, population: List[Prompt]) -> None:
        """Updates elitist, best_score_overall, best_prompt_overall.

        Args:
            population (List[Prompt]): current population.
        """
        scores = [prompt.score for prompt in population]
        best_score, best_sample_idx = max(scores), np.argmax(np.array(scores))

        if (
            self.best_score_overall is None
            or best_score >= self.best_score_overall
        ):
            self.best_score_overall = best_score
            self.best_prompt_overall = population[best_sample_idx].text
            self.elitist = population[best_sample_idx]
            logger.info(
                f"""Iteration {self.iteration}
                Elitist score: {self.best_score_overall}"""
            )
            logger.debug(f"Elitist text:\n{self.elitist.text}")

    def _update_iter(self, population: List[Prompt]) -> None:
        """Updates iteration. Cache current state.

        Args:
            population (List[Prompt]): current population.
        """
        logger.info(f"Iteration {self.iteration} finished...")
        logger.info(f"Best score: {self.best_score_overall}")

        population = self._reranking(population)
        self._cache_population(
            population, self._make_output_path("population")
        )

        self.iteration += 1

    def _long_term_reflection(self, short_term_reflections: List[str]) -> None:
        """Long-term reflection before mutation.

        Args:
            short_term_reflections (List[str]): short-term reflections.
        """
        request = self._long_term_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            PRIOR_LONG_TERM_REFLECTION=self._long_term_reflection_str,
            NEW_SHORT_TERM_REFLECTIONS="\n".join(short_term_reflections),
        )

        response = self._llm_query([request])[0]

        self._long_term_reflection_str = extract_answer(
            response, self.HINT_TAGS, format_mismatch_label=""
        )

    def _llm_query(self, requests: List[str]) -> List[str]:
        """Provides api to query requests to the model.

        Args:
            requests (List[str]): string requests.

        Returns:
            List[str]: model answers.
        """

        answers = self.model.batch(requests)

        return answers

    def _mutate(self) -> List[Prompt]:
        """Elitist-based mutation.

        Returns:
            List[Prompt]: generated population.
        """
        request = self._mutation_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            LONG_TERM_REFLECTION=self._long_term_reflection_str,
            ELITIST_PROMPT=self.elitist.text,
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

    def evolution(self) -> str:
        """Provides evolution operation.

        Selection -> Short-term reflection -> Long-term reflection
            -> Elitist-based mutation -> Survival.

        After all self.num_epochs epochs the best three prompts are selected.
        They will be evaluated on test split of dataset then.
        And based on their test scores,
        the best prompt will be returned.

        Returns:
            str: best evoluted prompt
        """

        population = np.array(self._init_pop())
        self._cache_population(
            population, self._make_output_path("initial_population.yaml")
        )

        while self.iteration < self.num_epochs:
            if self.elitist is not None and self.elitist not in population:
                logger.debug("Elitist should always live")
                population = np.append(population, np.array([self.elitist]))
            parent_population = self._selection(population)

            short_term_reflection_tuple = self._short_term_reflection(
                parent_population
            )
            self._cache_data(
                short_term_reflection_tuple[0],
                self._make_output_path("short_term_reflections"),
            )

            crossed_population = self._crossover(short_term_reflection_tuple)

            self._evaluation(crossed_population)
            self._update_elitist(crossed_population)

            self._long_term_reflection(short_term_reflection_tuple[0])
            self._cache_data(
                self._long_term_reflection_str,
                self._make_output_path("long_term_reflection"),
            )

            mutated_population = self._mutate()
            self._evaluation(mutated_population)

            population = np.append(population, np.array(crossed_population))
            population = np.append(population, np.array(mutated_population))
            self._update_elitist(population)
            population = self._survive(population, temperature=1e-1)

            self._update_iter(population)

        logger.info(f"BEST SCORE: {self.best_score_overall}")
        logger.debug(f"BEST PROMPT:\n{self.best_prompt_overall}")

        population = self._reranking(population)
        population = population[:4]
        population[3] = self.elitist
        self._evaluation(population, split="validation")
        self._cache_population(
            population, self._make_output_path("best_prompts_infer.yaml")
        )
        self._update_elitist(population)
        return self.elitist.text
