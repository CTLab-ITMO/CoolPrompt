"""
ZeroShot Baseline - простейший метод для сравнения.

Генерирует N случайных промптов (zero-order generation) и выбирает лучший на dev set.
Без эволюции, без адаптации - чистый baseline.

References:
    Используется как baseline в большинстве papers по prompt optimization.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rider.core.prompts import Prompt
from rider.core.operators import EvolutionaryOperators
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory

logger = logging.getLogger(__name__)


class ZeroShot:
    """
    ZeroShot Baseline: генерирует N случайных промптов и выбирает лучший.

    Алгоритм:
    1. Generate N prompts using zero-order generation
    2. Evaluate each on dev set
    3. Return best prompt

    Args:
        llm_client: LLM клиент для генерации промптов
        evaluator: PromptEvaluator для оценки промптов
        dataset_name: Название датасета
        num_prompts: Количество генерируемых промптов (default: 10)
        model: Модель для генерации (default: "gpt-3.5-turbo")
        temperature: Температура для генерации (default: 0.7)

    Example:
        >>> zeroshot = ZeroShot(
        ...     llm_client=llm,
        ...     evaluator=evaluator,
        ...     dataset_name='GSM8K',
        ...     num_prompts=10
        ... )
        >>> best_prompt = zeroshot.run(train_data, val_data, dev_data)
        >>> print(f"Best fitness: {best_prompt.fitness}")
    """

    def __init__(
        self,
        llm_client,
        evaluator: PromptEvaluator,
        dataset_name: str,
        num_prompts: int = 10,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        save_history: bool = True,
        log_detailed_evaluations: bool = True,
        experiment_name: str = None
    ):
        """Инициализация ZeroShot baseline."""
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.num_prompts = num_prompts
        self.model = model
        self.temperature = temperature
        self.save_history = save_history
        self.log_detailed_evaluations = log_detailed_evaluations
        self.experiment_name = experiment_name

        # Operators для zero-order generation
        self.operators = EvolutionaryOperators(
            llm_client=llm_client,
            model=model,
            temperature=temperature
        )

        # Evolution history для детального логирования
        if self.save_history:
            results_dir = Path("results")
            experiment_id = f"{dataset_name}_ZeroShot"
            # Use experiment_name as parent directory if provided
            if self.experiment_name:
                save_dir = results_dir / self.experiment_name / experiment_id
            else:
                save_dir = results_dir / experiment_id
            self.history = EvolutionHistory(
                save_dir=save_dir,
                experiment_id=experiment_id
            )
        else:
            self.history = None

        logger.info(
            f"ZeroShot initialized for {dataset_name}: "
            f"num_prompts={num_prompts}, model={model}"
        )

    def run(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        dev_data: List[Dict],
        test_data: Optional[List[Dict]] = None
    ) -> Prompt:
        """
        Запуск ZeroShot baseline.

        Args:
            train_data: Training data (не используется)
            val_data: Validation data (не используется)
            dev_data: Development data (не используется, оценка на val_data)
            test_data: Test data (optional)

        Returns:
            best_prompt: Лучший промпт с fitness на dev set
        """
        logger.info(f"Running ZeroShot on {self.dataset_name}...")
        logger.info(f"Generating {self.num_prompts} random prompts...")

        # Step 1: Generate N prompts using zero-order (parallel)
        task_desc = self._get_task_description()

        def _gen_prompt(i):
            prompt = self.operators.zero_order_generation(
                task_desc=task_desc,
                temperature=self.temperature
            )
            prompt.id = i
            return i, prompt

        prompts_dict = {}
        max_workers = min(32, self.num_prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_prompt, i): i for i in range(self.num_prompts)}
            for future in as_completed(futures):
                idx, prompt = future.result()
                prompts_dict[idx] = prompt

        prompts = [prompts_dict[i] for i in range(self.num_prompts)]

        logger.info(f"Generated {len(prompts)} prompts")

        # Step 2: Evaluate on val (same split as RIDER for fair comparison) — PARALLEL
        logger.info("Evaluating prompts on val set...")

        eval_results_map = {}

        def _eval_zs_prompt(idx_prompt):
            idx, p = idx_prompt
            try:
                if self.log_detailed_evaluations and self.history:
                    result = self.evaluator.evaluate_with_details(
                        prompt=p, dataset_name=self.dataset_name, data=val_data
                    )
                    metrics = result['metrics']
                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    p.fitness = metrics[primary_metric]
                    return idx, result
                else:
                    p.fitness = self.evaluator.evaluate_prompt(
                        prompt=p, dataset_name=self.dataset_name, data=val_data
                    )
                    return idx, None
            except Exception as e:
                logger.error(f"ZeroShot eval failed for prompt {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = min(8, len(prompts))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, result in executor.map(_eval_zs_prompt, enumerate(prompts)):
                eval_results_map[idx] = result

        # Sequential logging
        for i, prompt in enumerate(prompts):
            result = eval_results_map.get(i)
            if result and self.log_detailed_evaluations and self.history:
                predictions = result['predictions']
                ground_truth = result['ground_truth']
                metrics = result['metrics']
                primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                fitness = metrics[primary_metric]
                error_indices = [j for j, (pred, truth) in enumerate(zip(predictions, ground_truth)) if pred != truth]

                self.history.log_detailed_evaluation(
                    prompt_id=prompt.id,
                    generation=0,
                    dataset_name=self.dataset_name,
                    evaluation_details={
                        'fitness': fitness,
                        'predictions': predictions,
                        'ground_truth': ground_truth,
                        'error_indices': error_indices,
                        'metrics': metrics
                    }
                )

                self.history.log_evolution_step(
                    generation=0,
                    operator_used='zero_order_generation',
                    parent_ids=[],
                    parent_fitnesses=[],
                    offspring=prompt,
                    temperature=self.temperature,
                    top_p=1.0,
                    diversity_score=0.0,
                    accepted=True,
                    metadata={'candidate_num': i}
                )

        # Step 3: Select best
        best_prompt = max(prompts, key=lambda p: p.fitness)
        logger.info(f"Best prompt fitness: {best_prompt.fitness:.4f}")
        logger.info(f"Best prompt text: {best_prompt.text}")

        # Step 4: (Optional) Evaluate on test set
        if test_data:
            logger.info("Evaluating best prompt on test set...")
            best_prompt.test_fitness = self.evaluator.evaluate_prompt(
                prompt=best_prompt,
                dataset_name=self.dataset_name,
                data=test_data
            )
            logger.info(f"Test fitness: {best_prompt.test_fitness:.4f}")

        # Save history
        if self.save_history and self.history:
            self.history.save()
            logger.info("Evolution history saved")

        return best_prompt

    def _get_task_description(self) -> str:
        """
        Получить описание задачи для датасета.

        Returns:
            task_desc: Текстовое описание задачи
        """
        task_descriptions = {
            'GSM8K': 'Solve grade school math word problems step by step and provide the final numerical answer',
            'AG_News': 'Classify news articles into categories: World, Sports, Business, or Sci/Tech',
            'SQuAD_2': 'Answer questions based on the given context, or determine if the question is unanswerable',
            'CommonGen': 'Generate a coherent sentence that uses all the given concepts',
            'XSum': 'Generate a concise one-sentence summary of the given document',
        }

        return task_descriptions.get(
            self.dataset_name,
            f"Solve the task for dataset {self.dataset_name}"
        )
