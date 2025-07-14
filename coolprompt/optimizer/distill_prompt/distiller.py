"""Distiller module for prompt optimization.

This module provides the Distiller class for DistillPrompt optimization,
which handles the process of generating, evaluating, and refining prompts.
"""

import os
import yaml
from typing import Any, List

from tqdm import tqdm
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.utils.logging_config import logger
from coolprompt.optimizer.distill_prompt.candidate import (
    Candidate,
    CandidateHistory,
)
from coolprompt.optimizer.distill_prompt.generate import PromptTransformer
from coolprompt.optimizer.distill_prompt.utils import (
    TextSampler,
    seed_everything,
)


class Distiller:
    """Distiller class for DistillPrompt optimization.

    This class handles the process of optimizing prompts through
    multiple rounds of generation, evaluation, and refinement.

    Attributes:
        model: Language model to use for optimization.
        evaluator: Evaluator to compute metrics.
        train_dataset: Dataset to use while training.
        train_targets: Targets for train dataset.
        validation_dataset: Dataset to use while validating final prompts.
        validation_targets: Targets for validation dataset.
        base_prompt: Initial prompt to start optimization from.
        use_cache: Whether to cache intermediate results.
        num_epochs: Number of epochs to evaluate.
        output_path: Path to store logs of optimization.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        train_dataset: List[str],
        train_targets: List[str],
        validation_dataset: List[str],
        validation_targets: List[str],
        base_prompt: str,
        num_epochs: int = 10,
        output_path: str = "./distillprompt_outputs",
        use_cache: bool = True,
    ) -> None:
        """Initializes the Distiller with the given parameters.

        Args:
            model (BaseLanguageModel): Language model to use for optimization.
            evaluator (Evaluator): Evaluator to compute metrics.
            train_dataset (List[str]): Dataset to use while training.
            train_targets (List[str]): Targets for train dataset.
            validation_dataset (List[str]): Dataset to use while validating
                final prompts.
            validation_targets (List[str]): Targets for validation dataset.
            base_prompt (str): Initial prompt to start optimization from.
            num_epochs (int, optional): Number of epochs to evaluate.
                Defaults to 10.
            output_path (str, optional): Path to store logs of optimization.
                Defaults to './distillprompt_outputs'.
            use_cache (bool, optional): Whether to cache intermediate results.
                Defaults to True.
        """
        self.model = model
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets
        self.use_cache = use_cache
        self.base_prompt = base_prompt
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.iteration = 0
        self.logger = logger

        seed_everything()

    def _evaluate(self, prompt: str, split: str = "train") -> float:
        """Evaluates a given prompt on the specified dataset split.

        Args:
            prompt (str): The prompt to evaluate.
            split (str, optional): Dataset split to use
            ('train' or 'validation'). Defaults to 'train'.

        Returns:
            float: Evaluation score for the prompt.
        """
        if split == "train":
            dataset, targets = self.train_dataset, self.train_targets
        else:
            dataset = self.validation_dataset
            targets = self.validation_targets

        score = self.evaluator.evaluate(
            prompt=prompt,
            dataset=dataset,
            targets=targets,
        )
        return score

    def _cache_data(self, data: Any, savepath: os.PathLike) -> None:
        """Writes data to a YAML file if caching is enabled.

        Args:
            data (Any): Data to cache.
            savepath (os.PathLike): Path where to save the data.
        """
        if not self.use_cache:
            return

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w") as f:
            yaml.dump(data, f)

    def _make_output_path(self, filename: str) -> str:
        """Creates full path for logging based on current iteration.

        Args:
            filename (str): Base filename without extension.

        Returns:
            str: Full path including iteration number and extension.
        """
        return os.path.join(
            self.output_path, f"Iteration{self.iteration}", f"{filename}.yaml"
        )

    def distillation(self) -> str:
        """Performs DistillPrompt optimization.

        Executes the full optimization process through multiple rounds of
        generation, evaluation, and refinement of prompts.

        Returns:
            str: The best prompt found during optimization.
        """
        self.iteration = 0
        self.logger.info("Starting DistillPrompt optimization...")
        self.logger.debug(f"Start prompt:\n{self.base_prompt}")

        sampler = TextSampler(self.train_dataset, self.train_targets)
        transformer = PromptTransformer(self.model, sampler)
        history = CandidateHistory()

        base_prompt = self.base_prompt
        base_score = self._evaluate(base_prompt)
        base_candidate = Candidate(base_prompt, base_score)
        best_candidate = base_candidate

        for round_num in tqdm(range(self.num_epochs)):
            self.iteration = round_num + 1
            self.logger.info(f"Starting round {round_num}")
            history.clear()
            history.add(best_candidate)

            # Generation
            gen_prompts = transformer.generate_prompts(best_candidate)
            gen_candidates = [
                Candidate(prompt, self._evaluate(prompt))
                for prompt in gen_prompts
            ]
            history.extend(gen_candidates)

            # Distillation
            distilled_prompts = transformer.distill_samples(gen_candidates)
            distilled_candidates = [
                Candidate(prompt, self._evaluate(prompt))
                for prompt in distilled_prompts
            ]
            history.extend(distilled_candidates)

            # Compression
            compressed_prompts = transformer.compress_prompts(
                distilled_candidates
            )
            compressed_candidates = [
                Candidate(prompt, self._evaluate(prompt))
                for prompt in compressed_prompts
            ]
            history.extend(compressed_candidates)

            # Aggregation
            aggregated_prompt = transformer.aggregate_prompts(
                compressed_candidates
            )
            aggregated_candidate = Candidate(
                aggregated_prompt, self._evaluate(aggregated_prompt)
            )
            aggregated_synonyms = transformer.generate_synonyms(
                aggregated_candidate, n=3
            )

            final_candidates = [
                Candidate(prompt, self._evaluate(prompt))
                for prompt in aggregated_synonyms
            ]
            final_candidates.append(aggregated_candidate)
            history.extend(final_candidates)

            best_candidate = history.get_highest_scorer()
            self.logger.info(
                f"Best candidate score in round {round_num}: "
                f"{best_candidate.train_score}"
            )
            self.logger.debug(
                f"Best candidate prompt: {best_candidate.prompt}"
            )

            # Cache results
            self._cache_data(
                {
                    "prompts": [c.prompt for c in final_candidates],
                    "scores": [c.train_score for c in final_candidates],
                },
                self._make_output_path("round_results"),
            )

        final_prompt = best_candidate.prompt
        final_score = self._evaluate(final_prompt, split="validation")
        self.logger.info(
            f"Final best prompt score on validation: {final_score}"
        )
        self.logger.debug(f"Final best prompt: {final_prompt}")

        self._cache_data(
            {"final_prompt": final_prompt, "final_score": final_score},
            os.path.join(self.output_path, "final_results.yaml"),
        )

        self.logger.info("DistillPrompt optimization completed")

        return final_prompt
