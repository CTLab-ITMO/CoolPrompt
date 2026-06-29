from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import random
from time import sleep
from typing import Dict, List, Mapping, Optional, Any, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.evaluator import Evaluator
from coolprompt.language_model import create_chat_model
from coolprompt.optimizer.brave.actions import ActionResult
from coolprompt.optimizer.brave.batch_sampler import (
    StratifiedBatchSampler,
    CurriculumStratifiedBatchSampler,
)
from coolprompt.optimizer.brave.bayesian_sampling import StateFeaturizer
from coolprompt.optimizer.brave.controller import EVCController
from coolprompt.optimizer.brave.core_states import OptimizerState
from coolprompt.optimizer.brave.operation_logger import OperationLogger
from coolprompt.optimizer.brave.operators import (
    Operator,
    PopulationInitializationOperator,
    ParaphraseInitializationOperator,
    CrossoverOperator,
    ElitistMutationOperator,
    CompressorOperator,
    GradientStepOperator,
    HypeOperator,
    LongTermMutationOperator,
    ParaphrasingByPDOperator,
    ZeroOrderMutationOperator,
    CreativeRoleAndStyleMutationOperator,
    CreativeZeroOrderMutationOperator,
    HardFewShotExamplesOperator,
    BiggerPopulationInitializationOperator
)
from coolprompt.optimizer.brave.population_diversity import (
    PopulationDiversityManager
)
from coolprompt.optimizer.brave.utils import (
    BEGRAPEConfig,
    OptimizationLog,
    reranking_population
)
from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.utils.utils import get_dataset_split


class BEGRAPEEvoluter:
    """Budgeted Epistemic GRAPE (skeleton implementation).

    This class wires together:
    - EVC controller
    - drift-aware memory
    - budget-constrained loop
    """

    ACTIONS = {
        "crossover",
        "elitist_mutation",
        "compression",
        "gradient_step",
        "hype",
        "long_term_mutation",
        "paraphrase",
        "zero_order",
        "creative_role_and_style",
        "creative_zero_order",
        "few_shot_mutation"
    }

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        config: Optional[BEGRAPEConfig] = None,
        seed: int = 19,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self.logger = None
        if self.verbose and self.log_dir:
            self.logger = OperationLogger(log_dir=log_dir)

        self.model = create_chat_model(model)
        self.model.reset_stats()

        self.evaluator = evaluator
        self.cfg = config or BEGRAPEConfig()
        self.featurizer = StateFeaturizer()
        self.cost_tracker_stats = {}
        self.seed = seed
        self.train_batch_sampler: Optional[StratifiedBatchSampler] = None
        self.current_train_batch_data: Optional[List[str]] = None
        self.current_train_batch_targets: Optional[List[Any]] = None

        self.actions = self.cfg.actions
        if self.cfg.actions == "all":
            self.actions = list(self.ACTIONS)

        self.controller = EVCController(
            actions=self.actions,
            max_action_budget_share=self.cfg.max_action_budget_share,
            alpha_roi_ema=self.cfg.alpha_roi_ema,
            feature_dim=self.featurizer.dim,
            uncertainty_penalty_beta=self.cfg.uncertainty_penalty_beta,
            neural_weight=self.cfg.neural_weight,
            improve_prob_weight=self.cfg.improve_prob_weight,
            kill_switch_min_trials=self.cfg.kill_switch_min_trials,
            kill_switch_roi_threshold=self.cfg.kill_switch_roi_threshold,
            kill_switch_base_cooldown=self.cfg.kill_switch_base_cooldown,
            kill_switch_scaling_factor=self.cfg.kill_switch_scaling_factor,
            use_neural_bandit=self.cfg.use_neural_bandit,
            neural_hidden_dim=self.cfg.neural_hidden_dim,
            neural_learning_rate=self.cfg.neural_learning_rate,
            seed=seed,
        )

        self.diversity_manager = PopulationDiversityManager(
            similarity_threshold=self.cfg.diversity_similarity_threshold,
            max_per_cluster=self.cfg.diversity_max_per_cluster,
            auto_threshold=self.cfg.diversity_auto_threshold,
            target_cluster_count=self.cfg.population_size,
            use_hierarchical=self.cfg.diversity_use_hierarchical,
            use_bert=self.cfg.diversity_use_bert,
            bert_weight=self.cfg.diversity_bert_weight,
            duplicate_threshold=self.cfg.diversity_duplicate_threshold
        )

        self.long_term_reflection = ""
        self.short_term_reflections = []

        if self.cfg.population_initializer == "paraphrase":
            self.population_initializer = ParaphraseInitializationOperator(
                logger=self.logger
            )
        elif self.cfg.population_initializer == "begrape":
            self.population_initializer = PopulationInitializationOperator(
                logger=self.logger
            )
        elif self.cfg.population_initializer == "bigger":
            self.population_initializer = \
                BiggerPopulationInitializationOperator(
                    logger=self.logger
                )
        else:
            raise ValueError(
                "Unsupported population initializer: " +
                self.cfg.population_initializer
            )

        self.crossover_operator = CrossoverOperator(self.logger)
        self.elitist_mutation_operator = ElitistMutationOperator(self.logger)
        self.compressor_operator = CompressorOperator(
            model=self.model,
            logger=self.logger
        )
        self.gradient_step_operator = GradientStepOperator(self.logger)
        self.hype_operator = HypeOperator(model=self.model, logger=self.logger)
        self.long_term_mutation_operator = LongTermMutationOperator(self.logger)
        self.paraphrasing_operator = ParaphrasingByPDOperator(self.logger)
        self.zero_order_operator = ZeroOrderMutationOperator(self.logger)
        self.creative_role_and_style_operator =\
            CreativeRoleAndStyleMutationOperator(self.logger)
        self.creative_zero_order_mutation_operator =\
            CreativeZeroOrderMutationOperator(self.logger)

        self.population: List[Prompt] = []
        self.logs: List[OptimizationLog] = []
        self.initial_budget: float = self.cfg.initial_budget_tokens
        self.artifacts: Dict[str, bool] = self._init_artifacts()

    def _calculate_costs(
        self,
        new_tracker_stats: Dict[str, float]
    ) -> Dict[str, float]:
        delta = {
            metric: new_stat - self.cost_tracker_stats.get(metric, 0.0)
            for metric, new_stat in new_tracker_stats.items()
        }
        self.cost_tracker_stats = new_tracker_stats
        return delta

    @staticmethod
    def _init_artifacts() -> Dict[str, bool]:
        return {
            "has_eval": False,
            "has_failures": False,
            "has_gradients": False,
            "has_short_term": False,
            "has_offspring": False,
            "has_memory_update": False,
            "has_best_prompt": True,
        }

    def _update_artifacts(
        self,
        action: str,
        result: ActionResult,
        improved: bool
    ) -> None:
        if action == "crossover":
            self.artifacts["has_offspring"] = True
        if action == "mutation":
            self.artifacts["has_offspring"] = True
        if improved:
            self.artifacts["has_best_prompt"] = True

        payload_artifacts = result.payload.get("artifacts")
        if isinstance(payload_artifacts, Mapping):
            for k, v in payload_artifacts.items():
                if k in self.artifacts:
                    self.artifacts[k] = bool(v)

    def _compute_state(
        self,
        best_quality: float,
        recent_quality: List[float],
        useless_ops_count: int,
        steps_done: int,
        remaining_budget: float,
        population_diversity: float = 0.5,
    ) -> OptimizerState:
        slope = 0.0
        if len(recent_quality) >= 2:
            slope = max(min(recent_quality[-1] - recent_quality[-2], 1.0), -1.0)
        progress = min(steps_done / max(self.cfg.max_steps, 1), 1.0)
        useless_ratio = useless_ops_count / max(steps_done, 1)
        stagnation = 1.0 if abs(slope) < self.cfg.min_improvement else 0.0

        return OptimizerState(
            val_quality=float(np.clip(best_quality, 0.0, 1.0)),
            quality_slope=float(np.clip((slope + 1.0) / 2.0, 0.0, 1.0)),
            stagnation=float(stagnation),
            useless_ops_ratio=float(np.clip(useless_ratio, 0.0, 1.0)),
            remaining_budget_ratio=float(np.clip(
                remaining_budget / self.cfg.initial_budget_tokens, 0.0, 1.0
            )),
            epoch_progress=float(progress),
            population_diversity=float(np.clip(population_diversity, 0.0, 1.0)),
        )

    def _update_elitist(self, new_prompt: Prompt) -> float:
        mean_score = float(np.mean([p.score for p in self.population]))
        min_score = float(np.min([p.score for p in self.population]))
        delta_mean = new_prompt.score - mean_score
        delta_min = new_prompt.score - min_score
        delta = new_prompt.score - self.best_quality +\
            self.cfg.lambda_mean_quality * delta_mean +\
            self.cfg.lambda_min_quality * delta_min
        if new_prompt.score > self.best_quality:
            self.elitist = new_prompt
            self.best_quality = new_prompt.score
        elif new_prompt.score == self.best_quality \
                and self.elitist.origin == PromptOrigin.MANUAL:
            self.elitist = new_prompt
        return delta

    def _update_val_elitist(self, new_prompt: Prompt) -> None:
        vs = self._evaluate_val_cached(new_prompt)
        if vs > self.best_val_quality:
            self.best_val_quality = vs
            self.val_elitist = new_prompt

    def _rescore_population(self) -> None:
        for prompt in self.population:
            self._evaluate(prompt, split="train")
        self.population = reranking_population(self.population)
        self.elitist = self.population[0]
        self.best_quality = self.elitist.score

    def _add_new_prompt(self, prompt: Prompt, step: int) -> bool:
        self.population.append(prompt)
        self.population = reranking_population(self.population)
        if len(self.population) > self.cfg.population_size:
            if self.cfg.population_clusterization:
                self.population = self.diversity_manager.maintain_diversity(
                    self.population,
                    self.cfg.population_size
                )
                filter_report = self.diversity_manager.get_filter_report()
                if filter_report:
                    self.logger.log_diversity_filter(
                        step=step,
                        filter_report=filter_report
                    )
            else:
                self.population = self.population[:self.cfg.population_size]
        return prompt in self.population

    def _crossover(self, iteration: int) -> ActionResult:
        (
            offspring,
            short_term_reflection
        ) = self.crossover_operator.run(
            iteration=iteration,
            population=self.population,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )
        improved = self._add_new_prompt(offspring, iteration)
        delta_quality = self._update_elitist(offspring)

        self.short_term_reflections.append(short_term_reflection)
        if len(self.short_term_reflections) > self.cfg.population_size:
            self.short_term_reflections = self.short_term_reflections[1:]

        costs = self._calculate_costs(self.model.get_stats())
        return ActionResult(
            action="crossover",
            delta_quality=delta_quality,
            cost_tokens=costs['total_tokens'],
            improved=improved
        )

    def _elitist_mutation(self, iteration: int) -> ActionResult:
        prompt_to_mutate = self.elitist
        if random.random() < self.cfg.random_mutation_probability:
            prompt_to_mutate = np.random.choice(self.population)
        mutated, new_long_term_reflection = self.elitist_mutation_operator.run(
            iteration=iteration,
            elitist=prompt_to_mutate,
            problem_description=self.problem_description,
            long_term_reflection=self.long_term_reflection,
            short_term_reflections=self.short_term_reflections,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )
        improved = self._add_new_prompt(mutated, iteration)

        delta_quality = self._update_elitist(mutated)
        self.long_term_reflection = new_long_term_reflection
        costs = self._calculate_costs(self.model.get_stats())
        return ActionResult(
            action="mutation",
            delta_quality=delta_quality,
            cost_tokens=costs['total_tokens'],
            improved=improved
        )

    def _basic_mutation(
        self,
        iteration: int,
        action_name: str,
        mutation_operator: Operator,
        **kwargs
    ) -> ActionResult:
        prompt_to_mutate = np.random.choice(self.population)
        mutated = mutation_operator.run(
            iteration=iteration,
            prompt=prompt_to_mutate,
            **kwargs
        )
        improved = self._add_new_prompt(mutated, iteration)

        delta_quality = self._update_elitist(mutated)
        costs = self._calculate_costs(self.model.get_stats())
        return ActionResult(
            action=action_name,
            delta_quality=delta_quality,
            cost_tokens=costs['total_tokens'],
            improved=improved
        )

    def _compression(self, iteration: int) -> ActionResult:
        lengths = np.array([len(p.text) for p in self.population], dtype=float)
        weights = lengths / lengths.sum()
        ind = np.random.choice(len(self.population), p=weights)
        prompt_to_mutate = self.population[ind]
        mutated = self.compressor_operator.run(
            iteration=iteration,
            prompt=prompt_to_mutate,
            evaluate_fn=self._evaluate
        )
        improved = self._add_new_prompt(mutated, iteration)
        delta_quality = self._update_elitist(mutated)
        costs = self._calculate_costs(self.model.get_stats())
        return ActionResult(
            action="long_compression",
            delta_quality=delta_quality,
            cost_tokens=costs['total_tokens'],
            improved=improved
        )

    def _gradient_step(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="gradient_step",
            mutation_operator=self.gradient_step_operator,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _hype(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="hype",
            mutation_operator=self.hype_operator,
            problem_description=self.problem_description,
            evaluate_fn=self._evaluate
        )

    def _long_term_mutation(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="long_term_mutation",
            mutation_operator=self.long_term_mutation_operator,
            problem_description=self.problem_description,
            long_term_reflection=self.long_term_reflection,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _paraphrasing(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="paraphrase",
            mutation_operator=self.paraphrasing_operator,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _zero_order_mutation(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="zero_order",
            mutation_operator=self.zero_order_operator,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _creative_role_and_style_mutation(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="creative_role_and_style",
            mutation_operator=self.creative_role_and_style_operator,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _creative_zero_order_mutation(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="creative_zero_order",
            mutation_operator=self.creative_zero_order_mutation_operator,
            problem_description=self.problem_description,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _few_shots_mutation(self, iteration: int) -> ActionResult:
        return self._basic_mutation(
            iteration=iteration,
            action_name="few_shot_mutation",
            mutation_operator=self.few_shots_mutation_operator,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )

    def _execute_action(
        self,
        action: str,
        iteration: int
    ) -> ActionResult:
        match action:
            case "crossover": return self._crossover(iteration)
            case "elitist_mutation": return self._elitist_mutation(iteration)
            case "compression": return self._compression(iteration)
            case "gradient_step": return self._gradient_step(iteration)
            case "hype": return self._hype(iteration)
            case "long_term_mutation":
                return self._long_term_mutation(iteration)
            case "paraphrase": return self._paraphrasing(iteration)
            case "zero_order": return self._zero_order_mutation(iteration)
            case "creative_role_and_style":
                return self._creative_role_and_style_mutation(iteration)
            case "creative_zero_order":
                return self._creative_zero_order_mutation(iteration)
            case "few_shot_mutation": return self._few_shots_mutation(iteration)
            case _: raise ValueError(f"Unsupported action: {action}")

    def _init_train_batch_sampler(self) -> None:
        self.train_batch_sampler = None
        self.current_train_batch_data = None
        self.current_train_batch_targets = None
        self.current_train_batch_indices = None

        batch_size = int(self.cfg.train_batch_size)
        if not self.cfg.use_stratified_train_batches:
            return
        if batch_size <= 0:
            return
        if len(self.train_data) <= batch_size:
            return

        if self.cfg.use_curriculum_batches:
            self.train_batch_sampler = CurriculumStratifiedBatchSampler(
                task=self.evaluator.task,
                batch_size=batch_size,
                total_steps=self.cfg.max_steps,
                seed=self.seed,
                generation_bins=self.cfg.generation_strata_bins,
                warmup_steps=self.cfg.curriculum_warmup_steps,
                max_alpha=self.cfg.curriculum_max_alpha,
            )
        else:
            self.train_batch_sampler = StratifiedBatchSampler(
                task=self.evaluator.task,
                batch_size=batch_size,
                seed=self.seed,
                generation_bins=self.cfg.generation_strata_bins,
            )

    def _refresh_train_batch(self, epoch: int) -> None:
        if self.train_batch_sampler is None:
            self.current_train_batch_data = None
            self.current_train_batch_targets = None
            self.current_train_batch_indices = None
            return

        batch_indices = self.train_batch_sampler.sample(
            dataset=self.train_data,
            targets=self.train_targets,
            epoch=epoch,
        )
        self.current_train_batch_indices = batch_indices
        self.current_train_batch_data = [
            self.train_data[i] for i in batch_indices
        ]
        self.current_train_batch_targets = [
            self.train_targets[i] for i in batch_indices
        ]

    def _get_train_eval_data(self) -> Tuple[List[str], List[Any]]:
        if (
            self.current_train_batch_data is not None
            and self.current_train_batch_targets is not None
        ):
            return (
                self.current_train_batch_data,
                self.current_train_batch_targets
            )
        return self.train_data, self.train_targets

    def _evaluate_val_cached(self, prompt: Prompt) -> float:
        """Evaluate prompt on val set, using cached val_score if available.

        Does NOT overwrite prompt.score — train score is preserved.
        """
        if prompt.val_score is not None:
            return prompt.val_score
        try:
            score, _ = self.evaluator.evaluate(
                prompt=prompt.text,
                dataset=self.val_data,
                targets=self.val_targets,
                failed_examples=self.cfg.bad_examples_num,
            )
        except Exception:
            score = 0.0
        prompt.set_val_score(float(score))
        return prompt.val_score

    def _evaluate(self, prompt: Prompt, split="train") -> None:
        """Evaluates given prompt on self.dataset and records the score.

        Args:
            prompt (Prompt): a prompt to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        if split == "val":
            self._evaluate_val_cached(prompt)
            return
        dataset, targets = self._get_train_eval_data()

        try:
            score, bad_examples = self.evaluator.evaluate(
                prompt=prompt.text,
                dataset=dataset,
                targets=targets,
                failed_examples=self.cfg.bad_examples_num
            )
        except Exception:
            score = 0
            bad_examples = []

        prompt.set_score(score)
        prompt.set_bad_examples(bad_examples)

        if (
            split == "train"
            and isinstance(self.train_batch_sampler, CurriculumStratifiedBatchSampler)
            and self.current_train_batch_indices is not None
        ):
            bad_inputs = {ex['input'] for ex in bad_examples}
            failed_global = [
                self.current_train_batch_indices[i]
                for i, text in enumerate(dataset)
                if text in bad_inputs
            ]
            self.train_batch_sampler.update_difficulties(
                self.current_train_batch_indices,
                failed_global,
            )

    def _llm_query(self, requests: List[str]) -> List[str]:
        """Provides api to query requests to the model.

        Args:
            requests (List[str]): string requests.

        Returns:
            List[str]: model answers.
        """

        requests = [request.replace('\"', '\'') for request in requests]

        answers = None
        for _ in range(5):
            try:
                answers = self.model.batch(requests)
                break
            except Exception as e:
                print(e)
                sleep(60)

        if answers is None:
            return [""] * len(requests)

        answers = [a.content
                   if isinstance(a, AIMessage)
                   else a for a in answers]

        return answers

    def optimize(
        self,
        initial_prompt: str,
        problem_description: str,
        train_data: List[str],
        train_targets: List[str],
        val_data: List[str],
        val_targets: List[str],
    ) -> Dict[str, Any]:
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.artifacts = self._init_artifacts()
        remaining_budget = self.cfg.initial_budget_tokens
        self.initial_budget = self.cfg.initial_budget_tokens

        self.train_data = train_data
        self.train_targets = train_targets
        self.val_data = val_data
        self.val_targets = val_targets
        self.problem_description = problem_description

        needs_few_shot_data = (
            "few_shot_mutation" in self.actions
            or "hard_few_shot_mutation" in self.actions
        )
        if needs_few_shot_data:
            cnt = self.cfg.few_shot_examples_from_data_cnt
            ratio = cnt * 1.0 / len(self.train_data)
            max_num = self.cfg.few_shot_examples_max_num
            (
                self.train_data,
                examples_data,
                self.train_targets,
                examples_targets
            ) = get_dataset_split(
                dataset=self.train_data,
                target=self.train_targets,
                validation_size=ratio,
                train_as_test=False,
                random_state=self.seed,
            )
            data_sample = list(zip(examples_data, examples_targets))
            if "few_shot_mutation" in self.actions:
                self.few_shots_mutation_operator = HardFewShotExamplesOperator(
                    max_few_shot_examples_num=max_num,
                    data_sample=data_sample,
                    logger=self.logger
                )

        self._init_train_batch_sampler()
        self._refresh_train_batch(epoch=0)

        recent_quality: List[float] = [0.0]
        no_improve_steps = 0
        useless_ops_count = 0
        spent_tokens = 0.0
        self.val_elitist: Optional[Prompt] = None
        self.best_val_quality: float = 0.0

        self.population = self.population_initializer.run(
            initial_prompt=initial_prompt,
            population_size=self.cfg.initial_population_size,
            problem_description=problem_description,
            model=self.model,
            llm_query_fn=self._llm_query,
            evaluate_fn=self._evaluate
        )
        self.elitist = self.population[0]
        self.best_quality = self.elitist.score

        spent = self._calculate_costs(
            self.model.get_stats()
        )['total_tokens']
        remaining_budget -= spent
        spent_tokens += spent

        for step in range(1, self.cfg.max_steps + 1):
            if remaining_budget <= 0:
                break
            self._refresh_train_batch(epoch=step)

            if (
                self.cfg.rescore_steps > 0
                and step % self.cfg.rescore_steps == 0
                and self.train_batch_sampler is not None
            ):
                self._rescore_population()
                rescore_cost = self._calculate_costs(
                    self.model.get_stats()
                )['total_tokens']
                remaining_budget -= rescore_cost
                spent_tokens += rescore_cost

            diversity = self.diversity_manager.compute_diversity(self.population)
            state = self._compute_state(
                best_quality=self.best_quality,
                recent_quality=recent_quality,
                useless_ops_count=useless_ops_count,
                steps_done=step,
                remaining_budget=remaining_budget,
                population_diversity=diversity,
            )
            x = self.featurizer.transform(state)

            action, score_dict = self.controller.select_action(
                x=x,
                remaining_budget_tokens=remaining_budget,
                candidate_actions=self.actions,
            )
            if action is None:
                break

            action_score = float(score_dict.get(action, 0.0))
            is_fallback = "fallback" in score_dict
            controller_diag = {
                "action_score": action_score,
                "fallback": 1.0 if "fallback" in score_dict else 0.0,
                "ema_roi": float(
                    self.controller.action_stats[action]["ema_roi"]
                ),
                "trials": float(self.controller.action_stats[action]["trials"]),
            }

            self.logger.log_controller_state(
                iteration=step,
                selected_action=action,
                action_scores=score_dict,
                is_fallback=is_fallback,
                action_stats=self.controller.action_stats,
                global_step=self.controller.global_step
            )

            result = self._execute_action(action=action, iteration=step)

            # Update controller with realized outcome
            self.controller.update(
                action=action,
                x_before=x,
                delta_quality=result.delta_quality,
                actual_cost_tokens=result.cost_tokens,
                improved=result.improved,
            )

            # Apply token budget
            remaining_budget -= result.cost_tokens
            spent_tokens += result.cost_tokens

            # Update pseudo quality
            if result.improved:
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            useful_operation = result.delta_quality > 0.0
            if not useful_operation:
                useless_ops_count += 1

            self._update_artifacts(
                action=action,
                result=result,
                improved=result.improved
            )

            recent_quality.append(self.best_quality)
            value_per_token = result.delta_quality
            value_per_token /= max(result.cost_tokens, 1e-6)
            self.logs.append(
                OptimizationLog(
                    step=step,
                    action=action,
                    score=action_score,
                    delta_quality=result.delta_quality,
                    cost_tokens=result.cost_tokens,
                    cumulative_spent=spent_tokens,
                    value_per_token=value_per_token,
                    useful_operation=useful_operation,
                    controller_diag=controller_diag,
                    remaining_budget=max(remaining_budget, 0.0),
                    best_quality=self.best_quality,
                )
            )

            if step % 5 == 0:
                self.logger.log_population(step, self.population)

            if (
                self.cfg.val_checkpoint_steps > 0
                and step % self.cfg.val_checkpoint_steps == 0
            ):
                for candidate in self.population[:self.cfg.val_checkpoint_topk]:
                    self._update_val_elitist(candidate)

            if no_improve_steps >= self.cfg.patience_steps:
                break

            if self.cfg.early_stop and self.best_quality == 1.0:
                break

        self.logger.log_population(-2, self.population)

        for prompt in self.population:
            prompt.set_score(self._evaluate_val_cached(prompt))
            self._update_val_elitist(prompt)
        self.population.append(self.val_elitist)
        self.population = list(
            sorted(self.population, key=lambda prompt: prompt.val_score, reverse=True)
        )
        self.logger.log_population(-3, self.population)

        summary = {
            "best_prompt": self.elitist.text,
            "best_quality": self.best_quality,
            "best_val_prompt": self.val_elitist.text if self.val_elitist else self.elitist.text,
            "best_val_quality": self.best_val_quality if self.val_elitist else -1,
            "remaining_budget_tokens": max(remaining_budget, 0.0),
            "steps_done": len(self.logs),
            "logs": self.logs,
            "efficiency": self._build_efficiency_summary(),
            "controller": self.controller.diagnostics(),
            "artifacts": self.artifacts,
        }
        if self.log_dir and self.verbose:
            self.export_logs_jsonl(f"{self.log_dir}/all_iterations_log.jsonl")
            self.export_summary_json(
                f"{self.log_dir}/summary_log.json",
                summary
            )
        return summary

    def _quality_at_budget_fraction(self, fraction: float) -> float:
        target_spend = self.initial_budget * fraction
        best = 0.0
        for row in self.logs:
            if row.cumulative_spent <= target_spend:
                best = max(best, row.best_quality)
        return best

    def _build_efficiency_summary(self) -> Dict[str, Any]:
        if not self.logs:
            return {
                "spent_tokens": 0.0,
                "value_per_1k_tokens": 0.0,
                "useful_ops_ratio": 0.0,
                "quality_at_budget": {
                    "25%": 0.0,
                    "50%": 0.0,
                    "75%": 0.0,
                    "100%": 0.0
                },
            }

        spent_tokens = max(self.logs[-1].cumulative_spent, 1e-6)
        best_quality = self.logs[-1].best_quality
        useful_ops = sum(1 for x in self.logs if x.useful_operation)
        useful_ratio = useful_ops / max(len(self.logs), 1)
        return {
            "spent_tokens": spent_tokens,
            "value_per_1k_tokens": (best_quality / spent_tokens) * 1000.0,
            "useful_ops_ratio": useful_ratio,
            "quality_at_budget": {
                "25%": self._quality_at_budget_fraction(0.25),
                "50%": self._quality_at_budget_fraction(0.50),
                "75%": self._quality_at_budget_fraction(0.75),
                "100%": self._quality_at_budget_fraction(1.00),
            },
        }

    def export_logs_jsonl(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in self.logs:
                f.write(
                    json.dumps(
                        {
                            "step": row.step,
                            "action": row.action,
                            "score": row.score,
                            "delta_quality": row.delta_quality,
                            "cost_tokens": row.cost_tokens,
                            "cumulative_spent": row.cumulative_spent,
                            "value_per_token": row.value_per_token,
                            "useful_operation": row.useful_operation,
                            "controller_diag": row.controller_diag,
                            "remaining_budget": row.remaining_budget,
                            "best_quality": row.best_quality,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    def export_summary_json(
        self,
        path: str,
        summary: Optional[Dict[str, Any]] = None
    ) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = summary if summary is not None else {
            "best_quality": self.logs[-1].best_quality if self.logs else 0.0,
            "steps_done": len(self.logs),
            "efficiency": self._build_efficiency_summary(),
        }
        serializable = dict(payload)
        if "logs" in serializable:
            serializable["logs"] = [
                {
                    "step": row.step,
                    "action": row.action,
                    "score": row.score,
                    "delta_quality": row.delta_quality,
                    "cost_tokens": row.cost_tokens,
                    "cumulative_spent": row.cumulative_spent,
                    "value_per_token": row.value_per_token,
                    "useful_operation": row.useful_operation,
                    "controller_diag": row.controller_diag,
                    "remaining_budget": row.remaining_budget,
                    "best_quality": row.best_quality,
                }
                for row in serializable["logs"]
            ]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=True, indent=2)
