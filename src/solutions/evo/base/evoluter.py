from abc import ABC, abstractmethod
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re
import statistics
from typing import List, Any
import yaml
from langchain_core.language_models.base import BaseLanguageModel
from src.solutions.evo.base.prompt import Prompt, PromptOrigin
from coolprompt.evaluator import Evaluator


class Evoluter(ABC):
    """Basic evoluter class.
    Provides evolution interface.

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
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        train_dataset: List[str],
        train_target: List[str],
        validation_dataset: List[str],
        validation_target: List[str],
        evaluator: Evaluator,
        use_cache: bool = True,
    ) -> None:
        self._prompts_directory_path = './data/'
        self.model = model
        self.train_dataset = train_dataset
        self.train_target = train_target
        self.validation_dataset = validation_dataset
        self.validation_target = validation_target
        self.evaluator = evaluator
        self.use_cache = use_cache

        self.logger = logging.getLogger('Evoluter')
        self.logger.setLevel(logging.DEBUG)
        file_handler = TimedRotatingFileHandler(
            filename='evol.log', when="MIDNIGHT", interval=1, backupCount=30
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def _read_yaml_data(
        self,
        path: os.PathLike,
        key: str = 'prompts'
    ) -> Any:
        """Reads yaml file and provides data by key from it.

        Args:
            path (os.PathLike): path to yaml file.
            key (str, optional): A string key to extract data.
                Defaults to 'prompts'.

        Returns:
            Any: extracted data.
        """
        if not os.path.isfile(path):
            return {}

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data[key]

    def _reranking(self, population: List[Prompt]) -> List[Prompt]:
        """
        Sorts given population of prompts by their scores in descending order.

        Args:
            population (List[Prompt]): population to sort.

        Returns:
            List[Prompt]: sorted population.
        """
        return list(sorted(
            population, key=lambda prompt: prompt.score, reverse=True
        ))

    def _evaluate(self, prompt: Prompt, split="train") -> None:
        """Evaluates given prompt on self.dataset and records the score.

        Args:
            prompt (Prompt): a prompt to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        if split == "train":
            dataset, targets = self.train_dataset, self.train_target
        else:
            dataset, targets = self.validation_dataset, self.validation_target
        score = self.evaluator.evaluate(
            prompt=prompt.text,
            dataset=dataset,
            targets=targets,
        )
        prompt.set_score(score)

    def _evaluation(
        self,
        population: List[Prompt],
        split: str = 'train'
    ) -> None:
        """Evaluation operation for prompts population.
        Evaluates every prompt in population and records the results.

        Args:
            population (List[Prompt]): population of prompts to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        for prompt in population:
            self._evaluate(prompt, split=split)

    def _read_prompts(
        self,
        filename: str,
        origin: PromptOrigin = None
    ) -> List[Prompt]:
        """Creates prompt population from yaml file.

        Args:
            filename (str): a name of yaml file.
            origin (PromptOrigin, optional):
                can be used to override the existing prompt origin in file.
                Defaults to None.

        Returns:
            List[Prompt]: created population of prompts.
        """
        prompts_data = self._read_yaml_data(
            os.path.join(self._prompts_directory_path, filename)
        )
        return [
            Prompt.from_dict(prompt_data, origin=origin)
            for prompt_data in prompts_data
        ]

    def _init_pop(self, use_cache: bool) -> List[Prompt]:
        """Creates initial population of prompts.

        Args:
            use_cache (bool): whether to use caching or not.

        Returns:
            List[Prompt]: initial population.
        """
        if use_cache:
            cached_data = self._read_yaml_data(
                os.path.join(
                    self._prompts_directory_path,
                    'cached_prompts.yaml'
                )
            )

            if not cached_data:
                initial_population = self._init_pop(use_cache=False)
                self._cache_population(
                    initial_population,
                    os.path.join(
                        self._prompts_directory_path,
                        'cached_prompts.yaml'
                    )
                )
                return initial_population

            return [
                Prompt.from_dict(prompt_data) for prompt_data in cached_data
            ]

        initial_population = self._read_prompts(
            'prompts.yaml',
        )
        self._evaluation(initial_population)
        initial_population = self._reranking(initial_population)
        return initial_population

    def _cache_data(
        self,
        data: Any,
        savepath: os.PathLike
    ) -> None:
        """Writes the data to the yaml file.

        Args:
            data (Any): data to be cached.
            savepath (os.PathLike): a path to saving file.
        """
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as f:
            yaml.dump(data, f)

    def _cache_population(
        self,
        population: List[Prompt],
        savepath: os.PathLike
    ) -> None:
        """Caching a population of prompts to file.

        Args:
            population (List[Prompt]): prompt population.
            savepath (os.PathLike): a path to saving file.
        """
        best_score = population[0].score
        average_score = statistics.mean(
            [prompt.score for prompt in population]
        )
        data = {
            "best_score": best_score,
            "average_score": average_score,
            "prompts": [prompt.to_dict() for prompt in population]
        }
        self._cache_data(data, savepath)

    @abstractmethod
    def evolution(self) -> None:
        """Provides evolution operation."""
        pass

    @abstractmethod
    def _mutate(self, population: List[Prompt]) -> List[Prompt]:
        """Provides mutation operation.
        Generates new population of mutated prompts.

        Args:
            population (List[Prompt]): current prompt population.

        Returns:
            List[Prompt]: generated mutated prompt population.
        """
        pass

    @abstractmethod
    def _selection(
        self,
        population: List[Prompt]
    ) -> List[Prompt]:
        """Provides selection operation.

        Args:
            population (List[Prompt]): prompt population to select from.

        Returns:
            List[Prompt]: selected prompts.
        """
        pass
