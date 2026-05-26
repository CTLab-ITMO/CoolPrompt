"""
Детальное логирование и сохранение истории эволюции промптов.

Этот модуль собирает ВСЮHISTORY информацию о каждом шаге эволюции для
глубокого анализа: какие операторы работают, что улучшается/ухудшается,
как адаптируются гиперпараметры, и т.д.

КРИТИЧНО для понимания как улучшить RIDER!
"""

import json
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder для numpy типов (bool_, int64, float64 и т.д.)."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class PromptEvolutionStep:
    """
    Запись об одном шаге эволюции (создание нового промпта).

    Содержит ВСЮ информацию для анализа:
    - Какой оператор использовался
    - От каких родителей
    - Какой fitness был до/после
    - Какие гиперпараметры использовались
    - Был ли принят в популяцию или отклонен diversity filter
    """
    generation: int
    step_index: int  # Номер шага внутри поколения
    timestamp: float

    # Оператор и родители
    operator_used: str
    parent_ids: List[str]
    parent_fitnesses: List[float]

    # Созданный промпт
    offspring_id: str
    offspring_text: str
    offspring_fitness: float

    # Гиперпараметры
    temperature: float
    top_p: float
    diversity_score: float  # Diversity популяции в этот момент

    # Результат
    accepted: bool  # True если добавлен в популяцию, False если отклонен
    rejection_reason: Optional[str] = None  # "too_similar", "low_fitness", etc.

    # Improvement метрики
    best_parent_fitness: float = 0.0
    fitness_improvement: float = 0.0  # offspring_fitness - best_parent_fitness

    # Дополнительная информация
    metadata: Optional[Dict[str, Any]] = None

    # НОВОЕ: Evaluation details (для глубокого анализа ошибок)
    evaluation_data: Optional[Dict] = None  # Примеры на которых тестировали
    predictions: Optional[List[str]] = None  # Что предсказал промпт
    ground_truth: Optional[List[str]] = None  # Правильные ответы
    error_indices: Optional[List[int]] = None  # Индексы ошибочных примеров
    error_details: Optional[List[Dict]] = None  # Детали ошибок

    # НОВОЕ: Reflection insights
    reflection_output: Optional[str] = None  # Результат short-term reflection
    reflection_type: Optional[str] = None  # "short_term", "long_term", "lineage"

    # НОВОЕ: Parent prompts (для trajectory analysis)
    parent_texts: Optional[List[str]] = None  # Тексты родительских промптов

    # НОВОЕ: Memory context (если использовался)
    memory_patterns_used: Optional[List[str]] = None  # Паттерны из LongTermMemory

    def to_dict(self) -> Dict:
        """Преобразование в словарь для сериализации"""
        return asdict(self)


@dataclass
class GenerationSummary:
    """
    Сводка по одному поколению эволюции.

    Агрегирует всю информацию о поколении для quick analysis.
    """
    generation: int
    timestamp: float

    # Популяция
    population_size: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity_score: float

    # Elite
    elite_ids: List[str]
    elite_fitnesses: List[float]

    # Гиперпараметры
    avg_temperature: float
    avg_top_p: float

    # UCB статистика
    ucb_operator_distribution: Dict[str, float]  # % использования каждого оператора
    ucb_operator_success_rates: Dict[str, float]  # % успешных применений

    # Diversity statistics
    diversity_rejection_rate: float  # % промптов отклоненных из-за similarity
    avg_similarity: float  # Средняя cosine similarity в популяции

    # Evolution statistics
    total_offspring_created: int
    total_offspring_accepted: int
    improvement_over_previous: float  # best_fitness - previous_best_fitness

    # Long-term memory update (если было)
    memory_updated: bool = False
    memory_insights: Optional[str] = None

    # НОВОЕ: Elite trajectory (история лучшего промпта)
    elite_trajectory: Optional[List[Dict]] = None
    # [{generation: int, prompt_id: str, fitness: float, operator: str, parents: List[str]}]

    # НОВОЕ: Aggregated error analysis (агрегированный анализ ошибок)
    common_error_types: Optional[Dict[str, int]] = None  # Частые типы ошибок
    difficult_examples: Optional[List[Dict]] = None  # Самые сложные примеры

    # НОВОЕ: Reflection summary (сводка по рефлексии)
    reflection_insights_count: int = 0  # Сколько reflection вызовов
    top_reflection_insights: Optional[List[str]] = None  # Топ инсайтов

    # Hyperparameter details
    hyperparameter_stats: Optional[Dict] = None  # Full hyperparameter snapshot

    # Operator elimination tracking
    eliminated_operators: Optional[List[str]] = None  # All excluded operators this generation
    bayesian_eliminated: Optional[List[str]] = None  # Operators eliminated by Bayesian test (not hardcoded)

    # Timing
    generation_time: float = 0.0  # Время выполнения поколения в секундах

    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return asdict(self)


class EvolutionHistory:
    """
    Менеджер для сбора и сохранения полной истории эволюции.

    Собирает:
    - Каждый шаг эволюции (создание промпта)
    - Сводку по каждому поколению
    - UCB статистику
    - Hyperparameter адаптацию
    - Long-term memory инсайты

    Args:
        save_dir: Директория для сохранения истории
        experiment_id: Уникальный ID эксперимента

    Example:
        >>> history = EvolutionHistory(save_dir="results/", experiment_id="RIDER_GSM8K_run1")
        >>> # Логировать шаг эволюции
        >>> history.log_evolution_step(
        ...     generation=5,
        ...     operator="reflection_crossover",
        ...     parent_ids=["p_123", "p_456"],
        ...     offspring=prompt_object,
        ...     accepted=True
        ... )
        >>> # Логировать сводку поколения
        >>> history.log_generation_summary(
        ...     generation=5,
        ...     population=population_list,
        ...     ucb_stats=ucb_statistics
        ... )
        >>> # Сохранить всю историю
        >>> history.save()
    """

    def __init__(
        self,
        save_dir: Path,
        experiment_id: str,
        save_every_n_generations: int = 1,
        save_individual_prompts: bool = True
    ):
        """
        Инициализация менеджера истории.

        Args:
            save_dir: Директория для сохранения
            experiment_id: ID эксперимента
            save_every_n_generations: Автосохранение каждые N поколений
            save_individual_prompts: Сохранять ли полные тексты промптов
        """
        self.save_dir = Path(save_dir)
        self.experiment_id = experiment_id
        self.save_every_n_generations = save_every_n_generations
        self.save_individual_prompts = save_individual_prompts

        # Создать директории
        self.history_dir = self.save_dir / "history" / experiment_id
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # История
        self.evolution_steps: List[PromptEvolutionStep] = []
        self.generation_summaries: List[GenerationSummary] = []

        # Метаданные
        self.start_time = time.time()
        self.metadata = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().isoformat(),
            'save_individual_prompts': save_individual_prompts
        }

        logger.info(
            f"EvolutionHistory initialized for {experiment_id}. "
            f"Save dir: {self.history_dir}"
        )

    def load(self, path: Optional[Path] = None) -> None:
        """
        Загрузить историю из директории.

        Загружает:
        - evolution_steps.json: Шаги эволюции
        - generation_summaries.json: Сводки по поколениям
        - full_history.json: Метаданные

        Args:
            path: Путь к директории с историей. Если None, использует self.history_dir

        Raises:
            FileNotFoundError: Если директория не существует
        """
        load_dir = Path(path) if path else self.history_dir

        if not load_dir.exists():
            raise FileNotFoundError(f"History directory not found: {load_dir}")

        # Загрузить evolution_steps
        steps_file = load_dir / 'evolution_steps.json'
        if steps_file.exists():
            try:
                with open(steps_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.evolution_steps = [
                        PromptEvolutionStep(**step_data)
                        for step_data in data
                    ]
                logger.info(f"Loaded {len(self.evolution_steps)} evolution steps")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load evolution steps: {e}")

        # Загрузить generation_summaries
        summaries_file = load_dir / 'generation_summaries.json'
        if summaries_file.exists():
            try:
                with open(summaries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.generation_summaries = [
                        GenerationSummary(**summary_data)
                        for summary_data in data
                    ]
                logger.info(f"Loaded {len(self.generation_summaries)} generation summaries")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load generation summaries: {e}")

        # Загрузить метаданные из full_history
        full_history_file = load_dir / 'full_history.json'
        if full_history_file.exists():
            try:
                with open(full_history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', self.metadata)
                logger.info(f"Loaded metadata for experiment: {self.metadata.get('experiment_id', 'unknown')}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load metadata: {e}")

        # Обновить history_dir если загрузили из другого места
        if path:
            self.history_dir = load_dir

        logger.info(f"History loaded from {load_dir}")

    def log_evolution_step(
        self,
        generation: int,
        operator_used: str,
        parent_ids: List[str],
        parent_fitnesses: List[float],
        offspring,  # Prompt object
        temperature: float,
        top_p: float,
        diversity_score: float,
        accepted: bool,
        rejection_reason: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Логировать один шаг эволюции (создание промпта).

        Args:
            generation: Номер поколения
            operator_used: Название использованного оператора
            parent_ids: ID родительских промптов
            parent_fitnesses: Fitness родителей
            offspring: Созданный промпт (Prompt object)
            temperature: Использованная температура
            top_p: Использованный top-p
            diversity_score: Diversity популяции в этот момент
            accepted: Был ли промпт принят в популяцию
            rejection_reason: Причина отклонения (если accepted=False)
            metadata: Дополнительная информация
        """
        # Вычислить improvement
        best_parent_fitness = max(parent_fitnesses) if parent_fitnesses else 0.0
        fitness_improvement = offspring.fitness - best_parent_fitness

        step = PromptEvolutionStep(
            generation=generation,
            step_index=len(self.evolution_steps),
            timestamp=time.time(),
            operator_used=operator_used,
            parent_ids=parent_ids,
            parent_fitnesses=parent_fitnesses,
            offspring_id=offspring.id,
            offspring_text=offspring.text if self.save_individual_prompts else f"[ID: {offspring.id}]",
            offspring_fitness=offspring.fitness,
            temperature=temperature,
            top_p=top_p,
            diversity_score=diversity_score,
            accepted=accepted,
            rejection_reason=rejection_reason,
            best_parent_fitness=best_parent_fitness,
            fitness_improvement=fitness_improvement,
            metadata=metadata or {}
        )

        self.evolution_steps.append(step)

        logger.debug(
            f"Gen {generation}: Logged evolution step - "
            f"operator={operator_used}, "
            f"fitness_improvement={fitness_improvement:.4f}, "
            f"accepted={accepted}"
        )

    def log_generation_summary(
        self,
        generation: int,
        population: List,  # List[Prompt]
        elite_ids: List[str],
        ucb_stats: Dict,
        diversity_stats: Dict,
        hyperparameter_stats: Dict,
        timing: float,
        memory_updated: bool = False,
        memory_insights: Optional[str] = None,
        eliminated_operators: Optional[List[str]] = None,
        bayesian_eliminated: Optional[List[str]] = None
    ) -> None:
        """
        Логировать сводку по поколению.

        Args:
            generation: Номер поколения
            population: Текущая популяция
            elite_ids: ID элитных промптов
            ucb_stats: Статистика UCB селектора
            diversity_stats: Статистика diversity
            hyperparameter_stats: Статистика гиперпараметров
            timing: Время выполнения поколения
            memory_updated: Была ли обновлена long-term memory
            memory_insights: Insights из memory (если есть)
        """
        # Вычислить метрики популяции
        fitnesses = [p.fitness for p in population]
        elite_prompts = [p for p in population if p.id in elite_ids]

        # Improvement over previous generation
        if self.generation_summaries:
            prev_best = self.generation_summaries[-1].best_fitness
            improvement = max(fitnesses) - prev_best
        else:
            improvement = 0.0

        # Diversity rejection rate (из evolution_steps)
        gen_steps = [s for s in self.evolution_steps if s.generation == generation]
        if gen_steps:
            rejected_count = sum(1 for s in gen_steps if not s.accepted)
            diversity_rejection_rate = rejected_count / len(gen_steps)
        else:
            diversity_rejection_rate = 0.0

        summary = GenerationSummary(
            generation=generation,
            timestamp=time.time(),
            population_size=len(population),
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            avg_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            worst_fitness=min(fitnesses) if fitnesses else 0.0,
            diversity_score=diversity_stats.get('diversity_score', 0.0),
            elite_ids=elite_ids,
            elite_fitnesses=[p.fitness for p in elite_prompts],
            avg_temperature=hyperparameter_stats.get('avg_temperature', 0.7),
            avg_top_p=hyperparameter_stats.get('avg_top_p', 0.95),
            ucb_operator_distribution=ucb_stats.get('distribution', {}),
            ucb_operator_success_rates=ucb_stats.get('success_rates', {}),
            diversity_rejection_rate=diversity_rejection_rate,
            avg_similarity=diversity_stats.get('avg_similarity', 0.0),
            total_offspring_created=len(gen_steps),
            total_offspring_accepted=sum(1 for s in gen_steps if s.accepted),
            improvement_over_previous=improvement,
            memory_updated=memory_updated,
            memory_insights=memory_insights,
            hyperparameter_stats=hyperparameter_stats,
            eliminated_operators=eliminated_operators,
            bayesian_eliminated=bayesian_eliminated,
            generation_time=timing
        )

        self.generation_summaries.append(summary)

        logger.info(
            f"Gen {generation} summary: "
            f"best={summary.best_fitness:.3f}, "
            f"avg={summary.avg_fitness:.3f}, "
            f"diversity={summary.diversity_score:.3f}, "
            f"accepted={summary.total_offspring_accepted}/{summary.total_offspring_created}"
        )

        # Автосохранение
        if generation % self.save_every_n_generations == 0:
            self.save_checkpoint(generation)

    def log_detailed_evaluation(
        self,
        prompt_id: str,
        generation: int,
        dataset_name: str,
        evaluation_details: Dict,
    ) -> None:
        """
        Логирует детальные результаты evaluation.

        Args:
            prompt_id: ID промпта
            generation: Поколение
            dataset_name: Название датасета
            evaluation_details: Результат от evaluator.evaluate_with_details()
                {
                    'metrics': Dict,
                    'predictions': List[str],
                    'ground_truth': List[str],
                    'dataset_name': str,
                    'prompt_id': str
                }
        """
        predictions = evaluation_details['predictions']
        ground_truth = evaluation_details['ground_truth']
        metrics = evaluation_details['metrics']

        # Найти ошибки
        error_indices = []
        error_details = []
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if not self._is_correct(pred, truth, dataset_name):
                error_indices.append(i)
                error_details.append({
                    'index': i,
                    'prediction': pred,
                    'ground_truth': truth,
                    'error_type': self._classify_error(pred, truth, dataset_name)
                })

        # Подготовить данные для сохранения
        evaluation_log = {
            'prompt_id': prompt_id,
            'generation': generation,
            'dataset': dataset_name,
            'metrics': metrics,
            'total_examples': len(predictions),
            'correct_count': len(predictions) - len(error_indices),
            'error_count': len(error_indices),
            'error_rate': len(error_indices) / len(predictions) if predictions else 0,
            'error_details': error_details,  # Все ошибки
            'predictions_sample': predictions,  # Все predictions
            'ground_truth_sample': ground_truth,
            'timestamp': time.time()
        }

        # Сохранить в JSONL файл (evaluations_genXXX.jsonl)
        eval_file = self.history_dir / f"evaluations_gen{generation:03d}.jsonl"
        with open(eval_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation_log, ensure_ascii=False, cls=NumpyEncoder) + '\n')

        logger.debug(
            f"Logged detailed evaluation for prompt {str(prompt_id)[:8]}... "
            f"(gen {generation}): {len(error_indices)}/{len(predictions)} errors"
        )

    def _is_correct(self, prediction: str, ground_truth: str, dataset_name: str) -> bool:
        """
        Проверяет правильность предсказания.

        Args:
            prediction: Предсказание промпта
            ground_truth: Правильный ответ
            dataset_name: Название датасета

        Returns:
            True если предсказание правильное
        """
        # Нормализация
        pred = str(prediction).strip().lower()
        truth = str(ground_truth).strip().lower()

        # Для разных датасетов разная логика
        if dataset_name in ['GSM8K']:
            # Exact match для математики
            return pred == truth
        elif dataset_name in ['AG_News']:
            # Exact match для классификации
            return pred == truth
        elif dataset_name in ['SQuAD_2', 'CommonGen', 'XSum']:
            # Substring match для QA и генерации
            # (более мягкая проверка, но для детального анализа достаточно)
            return pred in truth or truth in pred
        else:
            # По умолчанию exact match
            return pred == truth

    def _classify_error(self, prediction: str, ground_truth: str, dataset_name: str) -> str:
        """
        Классифицирует тип ошибки.

        Args:
            prediction: Предсказание промпта
            ground_truth: Правильный ответ
            dataset_name: Название датасета

        Returns:
            Строка с типом ошибки
        """
        pred = str(prediction).strip()
        truth = str(ground_truth).strip()

        # Общие типы ошибок
        if not pred:
            return "empty_prediction"
        if len(pred) > len(truth) * 3:
            return "too_verbose"
        if len(pred) < len(truth) / 3:
            return "too_short"

        # Dataset-specific error types
        if dataset_name == 'GSM8K':
            # Математические ошибки
            if any(c.isdigit() for c in pred) and any(c.isdigit() for c in truth):
                return "wrong_number"
            return "wrong_answer"
        elif dataset_name == 'AG_News':
            # Ошибки классификации
            return "wrong_class"
        elif dataset_name in ['SQuAD_2', 'CommonGen', 'XSum']:
            # Ошибки генерации
            if pred[:20] == truth[:20]:
                return "partial_match"
            return "completely_wrong"
        else:
            return "unknown_error"

    def get_operator_analysis(self) -> Dict:
        """
        Анализ эффективности операторов на основе истории.

        Returns:
            Словарь с детальным анализом каждого оператора:
            {
                'operator_name': {
                    'total_uses': int,
                    'acceptance_rate': float,
                    'avg_fitness_improvement': float,
                    'success_rate': float (% улучшений fitness > 0),
                    'best_improvement': float,
                    'worst_improvement': float
                }
            }
        """
        operator_stats = defaultdict(lambda: {
            'uses': [],
            'improvements': [],
            'acceptances': []
        })

        for step in self.evolution_steps:
            op = step.operator_used
            operator_stats[op]['uses'].append(step)
            operator_stats[op]['improvements'].append(step.fitness_improvement)
            operator_stats[op]['acceptances'].append(1 if step.accepted else 0)

        # Агрегировать статистику
        analysis = {}
        for op, stats in operator_stats.items():
            total_uses = len(stats['uses'])
            improvements = stats['improvements']
            acceptances = stats['acceptances']

            analysis[op] = {
                'total_uses': total_uses,
                'acceptance_rate': sum(acceptances) / total_uses if total_uses > 0 else 0.0,
                'avg_fitness_improvement': sum(improvements) / total_uses if total_uses > 0 else 0.0,
                'success_rate': sum(1 for imp in improvements if imp > 0) / total_uses if total_uses > 0 else 0.0,
                'best_improvement': max(improvements) if improvements else 0.0,
                'worst_improvement': min(improvements) if improvements else 0.0
            }

        return analysis

    def get_generation_trends(self) -> Dict:
        """
        Анализ трендов по поколениям.

        Returns:
            Словарь с трендами:
            - fitness_trend: [список best fitness по поколениям]
            - diversity_trend: [список diversity scores]
            - temperature_trend: [список температур]
            - operator_usage_evolution: как менялось использование операторов
        """
        return {
            'fitness_trend': [s.best_fitness for s in self.generation_summaries],
            'avg_fitness_trend': [s.avg_fitness for s in self.generation_summaries],
            'diversity_trend': [s.diversity_score for s in self.generation_summaries],
            'temperature_trend': [s.avg_temperature for s in self.generation_summaries],
            'top_p_trend': [s.avg_top_p for s in self.generation_summaries],
            'improvement_trend': [s.improvement_over_previous for s in self.generation_summaries],
            'acceptance_rate_trend': [
                s.total_offspring_accepted / s.total_offspring_created if s.total_offspring_created > 0 else 0
                for s in self.generation_summaries
            ]
        }

    def track_elite_trajectory(self) -> List[Dict]:
        """
        Отслеживает полную траекторию лучшего промпта.

        Восстанавливает цепочку: init → mutation1 → mutation2 → ... → current_best

        Returns:
            Список шагов эволюции лучшего промпта с полной информацией:
            [
                {
                    'generation': int,
                    'prompt_id': str,
                    'prompt_text': str,
                    'fitness': float,
                    'operator': str,
                    'parents': List[str],
                    'improvement': float,
                    'reflection': Optional[str]
                },
                ...
            ]
        """
        # Найти текущий best prompt
        if not self.generation_summaries:
            return []

        last_gen = self.generation_summaries[-1]
        best_prompt_id = last_gen.elite_ids[0] if last_gen.elite_ids else None

        if not best_prompt_id:
            return []

        # Восстановить траекторию через parent_ids
        trajectory = []
        current_id = best_prompt_id

        # Защита от бесконечных циклов (на случай ошибок в данных)
        max_iterations = 1000
        iterations = 0

        while current_id and iterations < max_iterations:
            iterations += 1

            # Найти шаг где был создан этот промпт
            step = next((s for s in self.evolution_steps if s.offspring_id == current_id), None)
            if not step:
                break

            trajectory.insert(0, {
                'generation': step.generation,
                'prompt_id': current_id,
                'prompt_text': step.offspring_text,
                'fitness': step.offspring_fitness,
                'operator': step.operator_used,
                'parents': step.parent_ids,
                'improvement': step.fitness_improvement,
                'reflection': step.reflection_output if hasattr(step, 'reflection_output') else None
            })

            # Перейти к родителю (выбираем лучшего родителя)
            if step.parent_ids and step.parent_fitnesses:
                # Находим родителя с максимальным fitness
                parent_fitnesses = dict(zip(step.parent_ids, step.parent_fitnesses))
                current_id = max(parent_fitnesses, key=parent_fitnesses.get)
            else:
                # Нет родителей - это начальный промпт
                current_id = None

        return trajectory

    def analyze_failures(self, top_k: int = 10) -> Dict:
        """
        Анализирует частые типы ошибок и сложные примеры.

        Args:
            top_k: Количество топ примеров для вывода

        Returns:
            {
                'error_type_distribution': Dict[str, int],  # Частота типов ошибок
                'most_difficult_examples': List[Dict],  # Самые сложные примеры
                'operators_with_most_errors': List[Tuple[str, float]]  # Операторы с высоким % ошибок
            }
        """
        # Читаем все evaluations_*.jsonl файлы
        eval_files = list(self.history_dir.glob("evaluations_gen*.jsonl"))

        all_errors = []
        error_types = defaultdict(int)

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        eval_data = json.loads(line)
                        for error in eval_data.get('error_details', []):
                            all_errors.append(error)
                            error_types[error['error_type']] += 1
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Failed to read {eval_file}: {e}")

        # Найти самые сложные примеры (те на которых ошибались чаще всего)
        example_errors = defaultdict(list)
        for error in all_errors:
            key = error.get('ground_truth', '')
            example_errors[key].append(error)

        difficult_examples = sorted(
            [{'example': k, 'error_count': len(v), 'errors': v[:3]}
             for k, v in example_errors.items()],
            key=lambda x: x['error_count'],
            reverse=True
        )[:top_k]

        # Какие операторы чаще создают ошибочные промпты
        operator_errors = defaultdict(lambda: {'total': 0, 'errors': 0})

        # Вычислить средний fitness
        all_fitnesses = [s.offspring_fitness for s in self.evolution_steps]
        avg_fitness = np.mean(all_fitnesses) if all_fitnesses else 0.0

        for step in self.evolution_steps:
            operator_errors[step.operator_used]['total'] += 1
            # Считаем что промпт с fitness ниже среднего - "ошибочный"
            if step.offspring_fitness < avg_fitness:
                operator_errors[step.operator_used]['errors'] += 1

        operators_with_most_errors = sorted(
            [(op, stats['errors'] / stats['total'] if stats['total'] > 0 else 0)
             for op, stats in operator_errors.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'error_type_distribution': dict(error_types),
            'most_difficult_examples': difficult_examples,
            'operators_with_most_errors': operators_with_most_errors
        }

    def save_checkpoint(self, generation: int, population: Optional[list] = None) -> None:
        """
        Сохранить checkpoint истории.

        Args:
            generation: Текущее поколение
            population: Список промптов (для crash recovery)
        """
        checkpoint_file = self.history_dir / f"checkpoint_gen{generation:03d}.json"

        # Сериализуем популяцию для crash recovery
        pop_data = []
        if population:
            for p in population:
                pop_data.append({
                    'id': getattr(p, 'id', ''),
                    'text': getattr(p, 'text', ''),
                    'fitness': float(getattr(p, 'fitness', 0.0)),
                    'generation': getattr(p, 'generation', 0),
                    'mutation_type': getattr(p, 'mutation_type', ''),
                    'few_shot_examples': getattr(p, 'few_shot_examples', []),
                })

        checkpoint_data = {
            'metadata': self.metadata,
            'generation': generation,
            'generation_summaries': [s.to_dict() for s in self.generation_summaries],
            'operator_analysis': self.get_operator_analysis(),
            'trends': self.get_generation_trends(),
            'population': pop_data
        }

        # Сохранить evolution steps отдельно (может быть большим)
        if self.save_individual_prompts:
            steps_file = self.history_dir / f"evolution_steps_gen{generation:03d}.json"
            with open(steps_file, 'w') as f:
                json.dump([s.to_dict() for s in self.evolution_steps], f, indent=2, cls=NumpyEncoder)

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Saved checkpoint at generation {generation} to {checkpoint_file}")

    def save(self) -> None:
        """
        Сохранить полную историю.

        Создает несколько файлов:
        - full_history.json: Полная история
        - generation_summaries.json: Только сводки по поколениям
        - evolution_steps.json: Все шаги эволюции (если save_individual_prompts=True)
        - operator_analysis.json: Анализ операторов
        - trends.json: Тренды по поколениям
        """
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_time'] = time.time() - self.start_time
        self.metadata['total_generations'] = len(self.generation_summaries)
        self.metadata['total_evolution_steps'] = len(self.evolution_steps)

        # Сохранить сводки поколений
        summaries_file = self.history_dir / "generation_summaries.json"
        with open(summaries_file, 'w') as f:
            json.dump([s.to_dict() for s in self.generation_summaries], f, indent=2, cls=NumpyEncoder)

        # Сохранить шаги эволюции
        if self.save_individual_prompts:
            steps_file = self.history_dir / "evolution_steps.json"
            with open(steps_file, 'w') as f:
                json.dump([s.to_dict() for s in self.evolution_steps], f, indent=2, cls=NumpyEncoder)

        # Сохранить анализ операторов
        operator_analysis_file = self.history_dir / "operator_analysis.json"
        with open(operator_analysis_file, 'w') as f:
            json.dump(self.get_operator_analysis(), f, indent=2, cls=NumpyEncoder)

        # Сохранить тренды
        trends_file = self.history_dir / "trends.json"
        with open(trends_file, 'w') as f:
            json.dump(self.get_generation_trends(), f, indent=2, cls=NumpyEncoder)

        # Сохранить полную историю (все вместе)
        full_history_file = self.history_dir / "full_history.json"
        full_history = {
            'metadata': self.metadata,
            'generation_summaries': [s.to_dict() for s in self.generation_summaries],
            'operator_analysis': self.get_operator_analysis(),
            'trends': self.get_generation_trends()
        }
        with open(full_history_file, 'w') as f:
            json.dump(full_history, f, indent=2, cls=NumpyEncoder)

        logger.info(
            f"Saved full history for {self.experiment_id}. "
            f"{len(self.generation_summaries)} generations, "
            f"{len(self.evolution_steps)} evolution steps. "
            f"Location: {self.history_dir}"
        )

    def __repr__(self) -> str:
        """Строковое представление"""
        return (
            "EvolutionHistory("
            f"experiment={self.experiment_id}, "
            f"generations={len(self.generation_summaries)}, "
            f"steps={len(self.evolution_steps)})"
        )
