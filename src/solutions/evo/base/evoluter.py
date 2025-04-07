from abc import ABC, abstractmethod
import logging
import os
import statistics
from typing import List
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.solutions.evo.base.prompt import Prompt, PromptOrigin
from src.evaluation.evaluator import BaseNLPEvaluator
from src.data.base.datasets import BaseDataset


class Evoluter(ABC):

    def __init__(
        self,
        prompts_directory_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: BaseDataset,
        evaluator: BaseNLPEvaluator,
        metric: str,
        use_cache: bool = True,
    ) -> None:
        self._prompts_directory_path = prompts_directory_path
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.evaluator = evaluator
        self.metric = metric
        self.use_cache = use_cache

        self.logger = logging.getLogger('Evoluter')

    def _read_yaml_data(self, filename: str) -> List[dict]:
        path = os.path.join(self._prompts_directory_path, filename)
        if not os.path.isfile(path):
            self.logger.info(
                "Cache file will be created" +
                "after initial population evaluation"
            )
            return {}

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data['prompts']

    def _reranking(self, population: List[Prompt]) -> List[Prompt]:
        return list(sorted(population, key=lambda prompt: prompt.score))

    def _evaluation(self, population: List[Prompt]) -> None:
        for prompt in population:
            metrics = self.evaluator.evaluate_vllm(
                model=self.model,
                tokenizer=self.tokenizer,
                eval_ds=self.dataset
            )
            prompt.set_score(metrics[self.metric])

    def _read_prompts(
        self,
        filename: str,
        origin: PromptOrigin = None
    ) -> List[Prompt]:
        prompts_data = self._read_yaml_data(filename)
        return [
            Prompt.from_dict(prompt_data, origin=origin)
            for prompt_data in prompts_data
        ]

    def _init_pop(self, use_cache: bool) -> List[Prompt]:
        if use_cache:
            cached_data = self._read_yaml_data('cached_prompts.yaml')

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

        manual_prompts = self._read_prompts(
            'prompts.yaml',
            origin=PromptOrigin.manual
        )
        ape_prompts = self._read_prompts(
            'prompts_auto.yaml',
            origin=PromptOrigin.ape
        )
        initial_population = manual_prompts + ape_prompts
        initial_population = self._reranking(initial_population)
        return initial_population

    def _cache_population(
        self,
        population: List[Prompt],
        savepath: os.PathLike
    ) -> None:
        best_score = population[0].score
        average_score = statistics.mean(
            [prompt.score for prompt in population]
        )
        data = {
            "best_score": best_score,
            "average_score": average_score,
            "prompts": [prompt.to_dict() for prompt in population]
        }

        with open(savepath, 'w') as f:
            yaml.dump(data, f)

    @abstractmethod
    def evolution(self) -> None:
        pass

    @abstractmethod
    def _mutate(self, population: List[Prompt]) -> List[Prompt]:
        pass

    @abstractmethod
    def _selection(self, population: List[Prompt]) -> List[Prompt]:
        pass
