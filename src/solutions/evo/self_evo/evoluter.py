import os
from typing import List, Tuple
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import BaseNLPEvaluator
from src.solutions.evo.base import Evoluter, Prompt, PromptOrigin
from src.solutions.evo.self_evo.utils import parse_output, append_to_yaml


class ReEvoluter(Evoluter):
    """ReEvoluter class that represents basic evoluter with ReEvo ideas in it.

    Attributes:
        model: vllm.LLM class of model to use.
        tokenizer: PreTrainedTokenizer tokenizer to be used.
        dataset: a string name of dataset.
        evaluator: evaluator that implements BaseNLPEvaluator interface.
        metric: a string name of metric to optimize.
        task: a string name of task. Either ['cls' or 'gen']
        use_cache: a boolean variable.
            Either to use caching files for initial population or not.
        batch_size: an integer size of batch to use.
        population_size: an integer fixed size of prompt population.
        num_epochs: an integer number of epochs to evaluate.
        threshold: a float value to select individuals by their scores.
            It is used to put away bad individuals from parenting population.
        output_path: a path to store logs of evolution.
        elitist: a prompt with highest score in population.
        best_score_overall: best evaluation score during evolution.
        best_prompt_overall: text of prompt with best score overall.
        iteration: current iteration (epoch) of evolution.
        problem_description: string description of current promblem
            (that corresponds to dataset).
        
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        evaluator: BaseNLPEvaluator,
        metric: str,
        task: str,
        population_size: int = 10,
        num_epochs: int = 10,
        threshold: float = 0.0,
        output_path: str = './outputs',
        use_cache: bool = True,
        batch_size: int = 64,
    ) -> None:
        model = LLM(
            model=model_name,
            dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            evaluator=evaluator,
            metric=metric,
            task=task,
            use_cache=use_cache,
            batch_size=batch_size
        )

        self.population_size = population_size
        self.num_epochs = num_epochs
        self.config_filename = 'config_re.yaml'
        self.problem_description_filename = 'problems.yaml'
        self.output_path = output_path
        self.threshold = threshold

        self.elitist = None
        self._long_term_reflection_str = ""
        self.best_score_overall = None
        self.best_prompt_overall = None
        self.iteration = 0
        self._config_path = './data'

        self.problem_description = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.problem_description_filename
            ),
            key=self.dataset
        )
        self._short_term_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='short_term'
        )
        self._crossover_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='crossover'
        )
        self._long_term_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='long_term'
        )
        self._mutation_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='mutation'
        )

    def _selection(
        self,
        population: List[Prompt]
    ) -> List[Prompt]:
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
        # Eliminate invalid individuals
        population = [
            prompt for prompt in population if prompt.score > self.threshold
        ]
        if len(population) < 2:
            return None

        scores = np.array([prompt.score for prompt in population])
        probas = scores / np.sum(scores)

        trial = 0
        anyways = False
        while len(selected_population) < 2 * self.population_size:
            parents = np.random.choice(
                population,
                size=2,
                replace=False,
                p=probas
            )
            if parents[0].score != parents[1].score or anyways:
                selected_population.extend(parents)
            trial += 1
            if trial > 1000:
                anyways = True

        return selected_population

    def _survive(
        self,
        population: List[Prompt],
        temperature: float = None
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
            population,
            size=self.population_size,
            replace=False,
            p=probas
        )

    def _gen_short_term_reflection_prompt(
        self,
        ind1: Prompt,
        ind2: Prompt
    ) -> Tuple[str, str, str]:
        """Generates short-term reflection request into model.

        Args:
            ind1 (Prompt): first individual.
            ind2 (Prompt): second individual.

        Returns:
            Tuple[str, str, str]:
                string request, worse prompt text, better prompt text.
        """
        # Determine which individual is better or worse
        if ind1.score > ind2.score:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        request = self._short_term_template.replace(
            '<PROBLEM_DESCRIPTION>',
            self.problem_description
        ).replace(
            '<WORSE_PROMPT>',
            worse_ind.text
        ).replace(
            '<BETTER_PROMPT>',
            better_ind.text
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
            self.output_path,
            f"Iteration{self.iteration}",
            f"{filename}.yaml"
        )

    def _short_term_reflection(
        self,
        population: list[Prompt],
        verbose: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        """Short-term reflection before crossovering two individuals.

        Args:
            population (list[Prompt]): parenting population.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.

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

            (
                request,
                worse_prompt,
                better_prompt
            ) = self._gen_short_term_reflection_prompt(parent_1, parent_2)
            requests.append(request)
            worse_prompts.append(worse_prompt)
            better_prompts.append(better_prompt)

        responses = self._llm_query(requests, verbose=verbose)
        responses = [
            parse_output(response, bracket='<hint>') for response in responses
        ]
        return responses, worse_prompts, better_prompts

    def _crossover(
        self,
        short_term_reflection_tuple: Tuple[List[str], List[str], List[str]],
        verbose: bool = False
    ) -> List[Prompt]:
        """Provides crossover operation for ReEvo.

        Args:
            short_term_reflection_tuple
                (Tuple[List[str], List[str], List[str]]):
                    outputs of short-term reflection.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.

        Returns:
            List[Prompt]: new crossed prompts population.
        """
        (
            reflection_contents,
            worse_prompts,
            better_prompts
        ) = short_term_reflection_tuple
        requests = []
        for reflection, worse_prompt, better_prompt in zip(
            reflection_contents,
            worse_prompts,
            better_prompts
        ):
            request = self._crossover_template.replace(
                '<PROBLEM_DESCRIPTION>',
                self.problem_description
            ).replace(
                '<WORSE_PROMPT>',
                worse_prompt
            ).replace(
                '<BETTER_PROMPT>',
                better_prompt
            ).replace(
                '<SHORT_TERM_REFLECTION>',
                reflection
            )
            requests.append(request)

        responses = self._llm_query(requests, verbose=verbose)
        responses = [parse_output(response) for response in responses]
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
            or best_score > self.best_score_overall
        ):
            self.best_score_overall = best_score
            self.best_prompt_overall = population[best_sample_idx].text

        if self.elitist is None or best_score > self.elitist.score:
            self.elitist = population[best_sample_idx]
            self.logger.info(
                f"""Iteration {self.iteration}
                Elitist ({self.best_score_overall}):
                {self.elitist.text}"""
            )

    def _update_iter(self, population: List[Prompt]) -> None:
        """Updates iteration. Cache current state.

        Args:
            population (List[Prompt]): current population.
        """
        self.logger.info(f"Iteration {self.iteration} finished...")
        self.logger.info(f"Best score: {self.best_score_overall}")

        population = self._reranking(population)
        self._cache_population(
            population,
            self._make_output_path("population")
        )

        self.iteration += 1

    def _long_term_reflection(
        self,
        short_term_reflections: List[str],
        verbose: bool = False
    ) -> None:
        """Long-term reflection before mutation.

        Args:
            short_term_reflections (List[str]): short-term reflections.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.
        """
        request = self._long_term_template.replace(
            '<PROBLEM_DESCRIPTION>',
            self.problem_description
        ).replace(
            '<PRIOR_LONG_TERM_REFLECTION>',
            self._long_term_reflection_str
        ).replace(
            '<NEW_SHORT_TERM_REFLECTIONS>',
            '\n'.join(short_term_reflections)
        )

        response = self._llm_query(
            [request],
            verbose=verbose
        )[0]

        self._long_term_reflection_str = parse_output(
            response,
            bracket='<hint>'
        )

    def _llm_query(
        self,
        requests: List[str],
        verbose: bool = False,
        **config
    ) -> List[str]:
        """Provides api to query requests to the model.

        Args:
            requests (List[str]): string requests.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.
            config: additional sampling params.

        Returns:
            List[str]: model answers.
        """
        sampling_params = {
            "max_tokens": 150,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        sampling_params.update(**config)
        sampling_params = SamplingParams(**sampling_params)

        answers = self.model.generate(
            prompts=requests,
            sampling_params=sampling_params,
            use_tqdm=verbose
        )

        results = [answer.outputs[0].text for answer in answers]
        return results

    def _mutate(
        self,
        verbose: bool = False
    ) -> List[Prompt]:
        """Elitist-based mutation.

        Args:
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.

        Returns:
            List[Prompt]: generated population.
        """
        request = self._mutation_template.replace(
            '<PROBLEM_DESCRIPTION>',
            self.problem_description
        ).replace(
            '<LONG_TERM_REFLECTION>',
            self._long_term_reflection_str
        ).replace(
            '<ELITIST_PROMPT>',
            self.elitist.text
        )
        responses = self._llm_query(
            [request] * self.population_size,
            verbose=verbose,
            temperature=0.3
        )
        responses = [parse_output(response) for response in responses]
        population = [
            Prompt(response, origin=PromptOrigin.MUTATED)
            for response in responses
        ]
        return population

    def evolution(self) -> None:
        """Provides evolution operation.

        Selection -> Short-term reflection -> Long-term reflection
            -> Elitist-based mutation -> Survival.

        After all self.num_epochs epochs the best three prompts are selected.
        They will be evaluated on test split of dataset then.
        And based on their test scores,
        the best prompt will be written to the best_prompts.yaml.
        """
        population = np.array(self._init_pop(use_cache=self.use_cache))

        while self.iteration < self.num_epochs:
            if self.elitist is not None and self.elitist not in population:
                self.logger.info("Elitist should always live")
                population = np.append(population, np.array([self.elitist]))
            parent_population = self._selection(population)

            short_term_reflection_tuple = self._short_term_reflection(
                parent_population,
                verbose=True
            )
            self._cache_data(
                short_term_reflection_tuple[0],
                self._make_output_path("short_term_reflections")
            )

            crossed_population = self._crossover(
                short_term_reflection_tuple,
                verbose=True
            )

            self._evaluation(crossed_population)
            self._update_elitist(crossed_population)

            self._long_term_reflection(
                short_term_reflection_tuple[0],
                verbose=True
            )
            self._cache_data(
                self._long_term_reflection_str,
                self._make_output_path("long_term_reflection")
            )

            mutated_population = self._mutate(verbose=True)
            self._evaluation(mutated_population)

            population = np.append(population, np.array(crossed_population))
            population = np.append(population, np.array(mutated_population))
            self._update_elitist(population)
            population = self._survive(population, temperature=1e-1)

            self._update_iter(population)

        self.logger.info(f"BEST SCORE: {self.best_score_overall}")
        self.logger.info(f"BEST PROMPT:\n{self.best_prompt_overall}")

        population = self._reranking(population)
        population = population[:4]
        population[3] = self.elitist  # top-3 of current and elitist
        self._evaluation(population, split='test')
        self._cache_population(
            population,
            self._make_output_path('best_prompts_infer.yaml')
        )
        self._update_elitist(population)
        append_to_yaml(
            new_data={
                self.dataset: self.elitist.to_dict(),
            },
            filepath="./best_prompts.yaml"
        )
