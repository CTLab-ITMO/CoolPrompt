import os
from typing import List
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import BaseNLPEvaluator
from src.solutions.evo.base import Evoluter, Prompt
from src.solutions.evo.self_evo.utils import parse_output, append_to_yaml


class SPELLEvoluter(Evoluter):
    """
    SelfEvoluter class that represents basic evoluterwith ReEvo ideas in it.

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
        self.problem_description_filename = 'problems.yaml'
        self.config_filename = 'config.yaml'
        self.output_path = output_path
        self.iteration = 0
        self._config_path = './data'

        self.elitist = None
        self.best_score_overall = None
        self.best_prompt_overall = None

        self.problem_description = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.problem_description_filename
            ),
            key=self.dataset
        )
        self._crossmutation_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='crossmutation'
        )

    def _selection(
        self,
        population: List[Prompt],
        n: int,
    ) -> List[Prompt]:
        """Provides selection operation.
        Probabilities - softmax of scores.

        Args:
            population (List[Prompt]): prompt population to select from.
            n (int): number of individuals to select.

        Returns:
            List[Prompt]: selected prompts.
        """
        scores = np.array([prompt.score for prompt in population])
        probas = softmax(scores)

        return np.random.choice(
            population,
            size=n,
            replace=False,
            p=probas
        )

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

    def _create_parent_examples(self, parents: List[Prompt]) -> str:
        """Creates string of examples of parenting prompts.
        Each example is concatenation of text and score.

        Args:
            parents (List[Prompt]): parents to create examples

        Returns:
            str: resulting string
        """
        parents = [str(p).replace("\t", " ") for p in parents]
        return "\n".join(parents)

    def _mutate(
        self,
        population: List[Prompt],
        num_of_parents: int,
        verbose: bool = False
    ) -> List[Prompt]:
        """Crossover + mutation in one operation, described in SPELL article.

        Args:
            population (List[Prompt]): parenting population.
            num_of_parents (int):
                number of parents to base generation of a new prompt.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.

        Returns:
            List[Prompt]: generated population.
        """
        requests = []
        for i in range(0, len(population), num_of_parents):
            request = self._crossmutation_template.replace(
                '<PROBLEM_DESCRIPTION>',
                self.problem_description
            ).replace(
                '<EXAMPLES>',
                self._create_parent_examples(
                    population[i * num_of_parents:(i + 1) * num_of_parents]
                )
            )
            requests.append(request)

        responses = self._llm_query(
            requests,
            verbose=verbose,
            temperature=0.3
        )
        responses = [parse_output(response) for response in responses]
        population = [Prompt(response) for response in responses]
        return population

    def _parent_selection(
        self,
        population: List[Prompt],
        cnt: int,
        num_of_parents: int
    ) -> List[Prompt]:
        selected = np.array([])
        for _ in range(cnt):
            selected = np.append(
                selected,
                self._selection(population, num_of_parents)
            )
        return selected

    def evolution(self) -> None:
        """Provides evolution operation.

        Selection -> Mutation based on 1 parent -> Mutation based on 2 parents.
        All operations are described in SPELL article.

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

            parent_pop_1 = self._parent_selection(
                population,
                cnt=5,
                num_of_parents=1
            )
            mutated_pop_1 = self._mutate(
                parent_pop_1,
                num_of_parents=1,
                verbose=True
            )
            self._evaluation(mutated_pop_1)

            parent_pop_2 = self._parent_selection(
                population,
                cnt=5,
                num_of_parents=2
            )
            mutated_pop_2 = self._mutate(
                parent_pop_2,
                num_of_parents=2,
                verbose=True
            )
            self._evaluation(mutated_pop_2)

            population = np.append(population, np.array(mutated_pop_1))
            population = np.append(population, np.array(mutated_pop_2))
            self._update_elitist(population)
            population = self._selection(population, n=self.population_size)

            self._update_iter(population)

        self.logger.info(f"BEST SCORE: {self.best_score_overall}")
        self.logger.info(f"BEST PROMPT:\n{self.best_prompt_overall}")

        population = self._reranking(population)
        population = population[:4]
        population[3] = self.elitist  # top-3 of current and elitist
        self._evaluation(population, split='test')
        population = self._reranking(population)
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
