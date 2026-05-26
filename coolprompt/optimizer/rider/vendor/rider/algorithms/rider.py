"""
RIDER (Reflective Iterative Diversity-Enhanced Reasoning) - главный алгоритм.

Этот модуль реализует полный алгоритм RIDER со всеми компонентами:
- Windowed Thompson Sampling / UCB-based адаптивный выбор операторов
- 9 активных эволюционных операторов + VORTEX как stagnation-only escape
- Diversity management через SentenceTransformer
- Long-term memory, OPERATOR FORGE и GENESIS для накопления паттернов
- Error-directed evolution и рефлексия по ошибкам
- Task-adaptive приоритеты операторов
- CHIMERA few-shot co-evolution для generative задач
- PRISM + Racing для exact_match/F1 задач
- Adaptive hyperparameters через PHASE REACTOR

Архитектура:
1. Инициализация популяции (zero-order generation)
2. Эволюция через generations:
   - Оценка на validation
   - Thompson/UCB выбор оператора
   - Применение оператора с FORGE/GENESIS контекстом
   - PRISM/direct full eval в зависимости от метрики
   - (mu+lambda) selection с diversity tiebreak
   - Stagnation handling, CHIMERA, VORTEX
3. Финальный выбор лучшего prompt-а и сохранение истории
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
import copy
import glob as glob_module
import json
import math
import os
import random
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rider.core.prompts import Prompt
from rider.core.ucb_selector import UCBOperatorSelector
from rider.core.rider_operators import RIDEROperators
from rider.core.diversity import DiversityManager, kDPPSelector
from rider.core.memory import LongTermMemory
from rider.core.genesis import GenesisMemory
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory
from rider.config.task_priorities import get_task_operator_weights

logger = logging.getLogger(__name__)


class OperatorForge:
    """OPERATOR FORGE — operator self-memory.

    Each operator remembers its top-3 best outputs ever.
    When called again, these are injected as few-shot context.
    """

    def __init__(self, max_memories: int = 3):
        self.max_memories = max_memories
        self.memories: Dict[str, list] = {}  # {operator_name: [(prompt_text, fitness), ...]}

    def update(self, operator_name: str, prompt_text: str, fitness: float, population_median: float):
        """Store successful output if above population median."""
        if fitness <= population_median:
            return

        if operator_name not in self.memories:
            self.memories[operator_name] = []

        mem = self.memories[operator_name]
        mem.append((prompt_text, fitness))
        # Keep top-N by fitness
        mem.sort(key=lambda x: x[1], reverse=True)
        self.memories[operator_name] = mem[:self.max_memories]

    def get_context(self, operator_name: str) -> str:
        """Get few-shot context string for an operator."""
        mem = self.memories.get(operator_name, [])
        if not mem:
            return ""

        lines = ["\n\nHere are my best previous creations for this task:"]
        for text, fitness in mem:
            # Truncate long prompts
            short = text[:150] + "..." if len(text) > 150 else text
            lines.append(f"[Score: {fitness:.3f}] \"{short}\"")
        lines.append("\nCreate a new prompt that's even better.")
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, int]:
        """Return memory stats for logging."""
        return {op: len(mems) for op, mems in self.memories.items()}


class RIDER:
    """
    RIDER - Reflective Iterative Diversity-Enhanced Reasoning.

    Главный класс алгоритма RIDER, объединяющий все компоненты.

    Args:
        llm_client: LLM клиент для генерации промптов
        evaluator: PromptEvaluator для оценки промптов
        sentence_encoder: SentenceTransformer для embeddings
        dataset_name: Название датасета
        config: Конфигурация RIDER (dict с параметрами)
        model: Модель для генерации (default: "gpt-3.5-turbo")
        temperature: Базовая температура (default: 0.7)

    Example:
        >>> from rider.llm.client import LLMClient
        >>> from rider.evaluation.evaluator import PromptEvaluator
        >>> from rider.evaluation.metrics import MetricsEvaluator
        >>> from sentence_transformers import SentenceTransformer
        >>>
        >>> llm = LLMClient()
        >>> metrics = MetricsEvaluator()
        >>> evaluator = PromptEvaluator(llm, metrics)
        >>> encoder = SentenceTransformer("all-MiniLM-L6-v2")
        >>>
        >>> rider = RIDER(
        ...     llm_client=llm,
        ...     evaluator=evaluator,
        ...     sentence_encoder=encoder,
        ...     dataset_name='GSM8K',
        ...     config={'population_size': 15, 'num_generations': 12}
        ... )
        >>>
        >>> # Run evolution
        >>> best_prompt = rider.run(train_data, val_data, dev_data)
        >>> print(f"Best fitness: {best_prompt.fitness}")
    """

    # Автодетект типа задачи по метрике (вместо хардкода имён датасетов).
    # Руководитель: "отбрасывать операторов в процессе, не хардкодить по имени задачи"
    # bert_score_f1 → генеративная задача (summarization, generation)
    # exact_match → extractive (math, QA)
    # f1_macro → classification
    # f1 → QA с partial match
    _GENERATIVE_METRICS = {'bert_score_f1'}  # метрики генеративных задач
    _GENERATIVE_EXCLUDED = {'reflection_crossover'}  # слабый на генеративных (verbal gradients не работают)

    @classmethod
    def _is_generative_task(cls, dataset_name, evaluator=None):
        """автоматическое определение типа задачи по метрике."""
        if evaluator and hasattr(evaluator, 'metrics_evaluator'):
            metric = evaluator.metrics_evaluator.get_primary_metric_name(dataset_name)
            return metric in cls._GENERATIVE_METRICS
        # Fallback: известные датасеты (для обратной совместимости)
        return dataset_name in {'XSum', 'CommonGen', 'CodeSearchNet'}
    # first_order_refinement исключён — отрицательный avg improvement
    # во ВСЕХ 6 v9 экспериментах (-0.07 to -0.10). Добавляет жёсткие ограничения
    # (word count, tense, format rules) которые снижают fitness.
    _UNIVERSALLY_EXCLUDED = {'first_order_refinement'}

    # PHASE REACTOR — temperature and offspring modulation, not restriction.
    # All operators remain available. Historical phase boosts are retained in
    # config for compatibility/analysis, but are not applied in the main loop.
    # Phase 1 "IGNITION": broad exploration with generative operators.
    # Phase 2 "FUSION": error-driven refinement with analytical operators.
    # Phase 3 "CRYSTALLIZATION": polish with paraphrase and differential evolution.
    # PHASE REACTOR — temperature and offspring modulation, not restriction.
    # All operators remain available. Historical phase boosts are retained in
    # config for compatibility/analysis, but are not applied in the main loop.
    # Phase 1 "IGNITION": broad exploration with generative operators.
    # Phase 2 "FUSION": error-driven refinement with analytical operators.
    # Phase 3 "CRYSTALLIZATION": polish with paraphrase and differential evolution.
    # Boosts retained in config but NOT applied to UCB selector.
    # Empirical analysis showed boosts are WORSE than uniform Thompson Sampling.
    # Temperature modulation and offspring_mult remain active.
    _PHASE_CONFIG = {
        # Phase 1 "IGNITION": exploration boost
        0: {'name': 'IGNITION', 'boosts': {'zero_order': 2.0, 'eda_mutation': 1.5, 'eda_rank_index': 1.5, 'semantic_paraphrase': 1.0, 'contrastive_error_decomposition': 1.0}, 'temperature_mult': 1.3, 'offspring_mult': 3.0},
        1: {'name': 'IGNITION', 'boosts': {'zero_order': 2.0, 'eda_mutation': 1.5, 'eda_rank_index': 1.5, 'semantic_paraphrase': 1.0, 'contrastive_error_decomposition': 1.0}, 'temperature_mult': 1.2, 'offspring_mult': 3.0},
        # Phase 2 "FUSION": error-driven boost
        2: {'name': 'FUSION', 'boosts': {'contrastive_error_decomposition': 2.0, 'opro_trajectory_mutation': 1.5, 'eda_rank_index': 1.2, 'semantic_paraphrase': 1.0, 'eda_mutation': 1.0}, 'temperature_mult': 1.0, 'offspring_mult': 2.0},
        3: {'name': 'FUSION', 'boosts': {'contrastive_error_decomposition': 2.0, 'opro_trajectory_mutation': 1.5, 'eda_rank_index': 1.2, 'semantic_paraphrase': 1.0, 'eda_mutation': 1.0}, 'temperature_mult': 0.9, 'offspring_mult': 2.0},
        4: {'name': 'FUSION', 'boosts': {'contrastive_error_decomposition': 2.0, 'opro_trajectory_mutation': 1.5, 'semantic_paraphrase': 1.0, 'eda_mutation': 1.0}, 'temperature_mult': 0.9, 'offspring_mult': 2.0},
        # Phase 3 "CRYSTALLIZATION": polish boost
        5: {'name': 'CRYSTALLIZATION', 'boosts': {'semantic_paraphrase': 2.0, 'de_mutation': 1.5, 'opro_trajectory_mutation': 1.2, 'contrastive_error_decomposition': 1.0}, 'temperature_mult': 0.7, 'offspring_mult': 1.5},
        6: {'name': 'CRYSTALLIZATION', 'boosts': {'semantic_paraphrase': 2.0, 'de_mutation': 1.5, 'opro_trajectory_mutation': 1.0, 'contrastive_error_decomposition': 1.0}, 'temperature_mult': 0.6, 'offspring_mult': 1.5},
        7: {'name': 'CRYSTALLIZATION', 'boosts': {'semantic_paraphrase': 2.0, 'de_mutation': 1.5, 'contrastive_error_decomposition': 1.0}, 'temperature_mult': 0.5, 'offspring_mult': 1.5},
    }

    def __init__(
        self,
        llm_client,  # LLMClient
        evaluator: PromptEvaluator,
        sentence_encoder,  # SentenceTransformer
        dataset_name: str,
        config: Dict,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        experiment_name: str = None,
        cross_experiment_memory: Optional[bool] = None
    ):
        """
        Инициализация RIDER.

        Args:
            llm_client: LLM клиент
            evaluator: Evaluator для оценки промптов
            sentence_encoder: Encoder для diversity
            dataset_name: Название датасета
            config: Конфигурация (population_size, num_generations, etc.)
            model: Модель для генерации
            temperature: Базовая температура
        """
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.sentence_encoder = sentence_encoder
        self.dataset_name = dataset_name
        self.config = config
        self.model = model
        self.temperature = temperature

        # Cross-Experiment Memory loads best prompts from previous RIDER runs on the
        # same dataset as warm start. This is a RIDER-unique contribution. For fair
        # comparison with baselines in publications, set cross_experiment_memory=False
        # to disable this feature and run RIDER "cold start" equivalent to baselines.
        self.cross_experiment_memory = (
            cross_experiment_memory
            if cross_experiment_memory is not None
            else config.get('cross_experiment_memory', True)
        )

        # ========== Инициализация компонентов ==========

        # Long-term memory
        self.long_term_memory = LongTermMemory(
            max_patterns=config.get('max_memory_patterns', 20)
        )

        # RIDER operators (с memory)
        self.operators = RIDEROperators(
            llm_client=llm_client,
            sentence_encoder=sentence_encoder,
            long_term_memory=self.long_term_memory,
            model=model,
            temperature=temperature
        )

        # Для генеративных задач сохраняем CoT scaffolding в промптах.
        # автодетект по метрике вместо хардкода имён.
        if self._is_generative_task(dataset_name, evaluator):
            self.operators.preserve_cot = True
            logger.info(f"preserve_cot=True for {dataset_name}")

        # Мягкий diversity filter — threshold 0.95 (было 0.65-0.80).
        # Старый threshold отклонял 75-85% offspring, убивая эволюцию.
        # Threshold 0.95 пропускает ~95%+ offspring, отклоняя только почти идентичные.
        self.diversity_manager = DiversityManager(
            sentence_encoder=sentence_encoder,
            diversity_threshold=0.95,
            adaptive_threshold=False,  # Фиксированный при 0.95
            min_threshold=0.95,
            max_threshold=0.95
        )

        # k-DPP selector для ensemble
        self.kdpp_selector = kDPPSelector()

        # UCB selector с task-adaptive приоритетами
        # 9 активных операторов.
        # Удалены (устаревшее / неэффективное): ga_crossover (2%), lamarckian (3%),
        # lineage_based (-42%), elitist_mutation (0%), concept_brainstorm (0%),
        # short_term_reflection (полностью удалён в v18.1.2),
        # first_order_refinement, prompt_paraphrase, prompt_improvement.
        # _UNIVERSALLY_EXCLUDED guard всё ещё перенаправляет first_order_refinement -> zero_order
        # для обратной совместимости со старыми checkpoints.
        self.available_operators = [
            'eda_mutation', 'eda_rank_index', 'zero_order',
            'de_mutation', 'ga_mutation',
            'reflection_crossover',
            'opro_trajectory_mutation', 'contrastive_error_decomposition', 'semantic_paraphrase'
        ]

        # Task-adaptive приоритеты операторов (из task_priorities.py)
        task_weights = get_task_operator_weights(dataset_name)

        self.ucb_selector = UCBOperatorSelector(
            operators=self.available_operators,
            c=config.get('ucb_c', 1.414),  # √2
            use_thompson_sampling=config.get('use_thompson_sampling', True),
            initial_rewards=task_weights
        )

        # Evolution history (для детального логирования)
        results_dir = Path(config.get('results_dir', './results'))
        experiment_id = f"{dataset_name}_{self.__class__.__name__}"
        # Use experiment_name as parent directory if provided
        if experiment_name:
            history_dir = results_dir / experiment_name / experiment_id
        else:
            history_dir = results_dir / experiment_id
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history = EvolutionHistory(
            save_dir=history_dir,
            experiment_id=experiment_id
        )

        # ========== Состояние алгоритма ==========

        self.population: List[Prompt] = []
        self.best_prompt: Optional[Prompt] = None
        self.elite_history: List[Prompt] = []

        # ========== Stagnation Escape ==========
        self.base_temperature = temperature
        self.current_temperature = temperature
        self.current_top_p = 0.95
        self.stagnation_fitness_history: List[float] = []
        self.stagnation_count = 0  # Consecutive gens without > 1% improvement
        self._restart_count = 0  # Early stop also tracks accumulated soft restarts.
        self.current_errors: List[Dict] = []  # Error examples from best prompt

        # OPERATOR FORGE — operator self-memory
        self.operator_forge = OperatorForge(max_memories=3)

        # GENESIS — ancestral lesson memory
        self.genesis = GenesisMemory(max_lessons=5)

        # Флаги
        self.use_pareto = config.get('use_pareto_selection', False)

        logger.info(
            f"RIDER initialized for {dataset_name}: "
            f"pop={config.get('population_size')}, "
            f"gen={config.get('num_generations')}, "
            f"pareto={self.use_pareto}"
        )

    # ========== Warm Start ==========

    @staticmethod
    def load_population_from_checkpoint(prev_exp_dir: Path, dataset_name: str) -> List[Prompt]:
        """
        Загружает финальную популяцию промптов из предыдущего эксперимента.

        Читает evolution_steps_gen{N}.json (последний) и возвращает все промпты,
        отсортированные по fitness (лучшие первые).

        Args:
            prev_exp_dir: Путь к директории предыдущего эксперимента
            dataset_name: Название датасета

        Returns:
            Список Prompt объектов (финальная популяция)
        """
        prev_exp_dir = Path(prev_exp_dir)
        experiment_id = f"{dataset_name}_RIDER"
        history_dir = prev_exp_dir / experiment_id / "history" / experiment_id

        if not history_dir.exists():
            logger.warning(f"Warm start: history dir not found: {history_dir}")
            return []

        # Собираем все промпты из всех файлов evolution_steps
        all_prompts: Dict[str, Prompt] = {}
        step_files = sorted(history_dir.glob("evolution_steps_gen*.json"))

        if not step_files:
            logger.warning(f"Warm start: no evolution_steps files in {history_dir}")
            return []

        for step_file in step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    steps = json.load(f)
                for step in steps:
                    pid = step.get("offspring_id", "")
                    text = step.get("offspring_text", "")
                    fitness = step.get("offspring_fitness", 0.0)
                    generation = step.get("generation", 0)
                    if pid and text:
                        # Обновляем если встретился лучший fitness для того же ID
                        if pid not in all_prompts or fitness > all_prompts[pid].fitness:
                            all_prompts[pid] = Prompt(
                                text=text,
                                fitness=fitness,
                                id=f"warmstart_{pid}",
                                generation=generation
                            )
            except Exception as e:
                logger.warning(f"Warm start: failed to read {step_file}: {e}")

        if not all_prompts:
            logger.warning(f"Warm start: no prompts found in {history_dir}")
            return []

        # Сортируем по fitness (лучшие первые)
        sorted_prompts = sorted(all_prompts.values(), key=lambda p: p.fitness, reverse=True)
        logger.info(
            f"Warm start: loaded {len(sorted_prompts)} prompts from {prev_exp_dir.name} "
            f"(best fitness: {sorted_prompts[0].fitness:.4f})"
        )
        return sorted_prompts

    @staticmethod
    def load_crash_recovery_checkpoint(
        experiment_dir: Path, dataset_name: str
    ) -> tuple:
        """
        Load population + generation from latest checkpoint for crash recovery.

        Returns:
            (population, start_generation) or (None, 0) if no checkpoint found
        """
        experiment_dir = Path(experiment_dir)
        experiment_id = f"{dataset_name}_RIDER"
        history_dir = experiment_dir / experiment_id / "history" / experiment_id

        if not history_dir.exists():
            return None, 0

        # Find latest checkpoint with population
        checkpoint_files = sorted(history_dir.glob("checkpoint_gen*.json"))
        if not checkpoint_files:
            return None, 0

        # Try from latest to earliest
        for cp_file in reversed(checkpoint_files):
            try:
                with open(cp_file, 'r', encoding='utf-8') as f:
                    cp_data = json.load(f)

                pop_data = cp_data.get('population', [])
                gen = cp_data.get('generation', 0)

                if not pop_data:
                    continue

                # Reconstruct population
                population = []
                for p in pop_data:
                    prompt = Prompt(
                        text=p['text'],
                        fitness=p.get('fitness', 0.0),
                        id=p.get('id', ''),
                        generation=p.get('generation', 0),
                        mutation_type=p.get('mutation_type', ''),
                        few_shot_examples=p.get('few_shot_examples', [])
                    )
                    population.append(prompt)

                if population:
                    logger.info(
                        f"Crash recovery: loaded {len(population)} prompts from "
                        f"checkpoint Gen {gen} (best={max(p.fitness for p in population):.4f})"
                    )
                    # Resume from NEXT generation
                    return population, gen + 1

            except Exception as e:
                logger.warning(f"Crash recovery: failed to read {cp_file}: {e}")
                continue

        return None, 0

    # ========== Data-Aware Seeding ==========

    def _format_demos_for_seeding(
        self, train_data: List[Dict], dataset_name: str
    ) -> str:
        """
        Форматирует train данные для APE-style seeding.

        Показывает LLM реальные input/output пары, чтобы он сам вывел инструкцию.
        Это тот же подход что у APE (Zhou 2022), адаптированный для RIDER seeding.

        Args:
            train_data: Примеры из train_data (5 штук)
            dataset_name: Название датасета

        Returns:
            Отформатированная строка с демо-примерами
        """
        demos = []
        for i, ex in enumerate(train_data[:5], 1):
            if dataset_name == 'GSM8K':
                demos.append(f"Input {i}: {ex['question']}\nOutput {i}: {ex['answer']}")
            elif dataset_name == 'AG_News':
                demos.append(f"Input {i}: {ex['text'][:200]}\nOutput {i}: {ex['label']}")
            elif dataset_name == 'SQuAD_2':
                ctx = ex.get('context', '')[:300]
                q = ex.get('question', '')
                ans = ex.get('answers', [''])[0] if ex.get('answers') else ''
                demos.append(f"Input {i}: {ctx}... Question: {q}\nOutput {i}: {ans}")
            elif dataset_name == 'CommonGen':
                concepts = ', '.join(ex.get('concepts', []))
                demos.append(f"Input {i}: {concepts}\nOutput {i}: {ex.get('target', '')}")
            elif dataset_name == 'XSum':
                demos.append(f"Input {i}: {ex.get('document', '')[:300]}...\nOutput {i}: {ex.get('summary', '')}")
            else:
                out = ex.get('output', ex.get('label', ex.get('answer', '')))
                demos.append(
                    f"Input {i}: {str(ex.get('input', ex.get('text', ex.get('question', str(ex)[:200]))))[:200]}\n"
                    f"Output {i}: {str(out)}"
                )
        return "\n\n".join(demos)

    def _generate_data_aware_prompt(
        self, train_data: List[Dict], dataset_name: str, task_desc: str,
        temperature: float = 1.0
    ) -> Prompt:
        """
        APE-style data-aware prompt generation.

        Показывает LLM реальные input/output пары и просит: "какая инструкция
        это создала?". Для XSum LLM увидит обрезанные статьи + первые предложения
        и выведет: "это задача восстановления lead sentence" → fitness 0.43+.

        Args:
            train_data: Примеры из train_data
            dataset_name: Название датасета
            task_desc: Описание задачи
            temperature: Температура генерации (разные T для разнообразия)

        Returns:
            Новый Prompt на основе данных
        """
        demos = self._format_demos_for_seeding(train_data, dataset_name)

        ape_prompt = f"""I gave a friend an instruction and some inputs. The friend read the instruction and wrote an output for every one of the inputs.
Here are the input-output pairs:

{demos}

What could the instruction have been?
Be specific and detailed (50-100 words). Include output format rules, edge cases, and constraints.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            from rider.core.operators import extract_prompt_from_response

            response = self.llm_client.generate(
                prompt=ape_prompt,
                model=self.model,
                temperature=temperature,
                max_tokens=350,
                top_p=0.99
            )

            text = extract_prompt_from_response(response, self.operators.preserve_cot)

            # Валидация
            if not self.operators._validate_prompt_text(text):
                logger.warning("Data-aware prompt validation failed, using fallback")
                return self.operators.zero_order_generation(
                    task_desc, dataset_name=dataset_name
                )

            return Prompt(
                text=text,
                generation=0,
                mutation_type="data_aware_seeding",
                metadata={'source': 'ape_style'}
            )

        except Exception as e:
            logger.error(f"Data-aware seeding error: {e}")
            return self.operators.zero_order_generation(
                task_desc, dataset_name=dataset_name
            )

    # ========== Cross-Experiment Memory ==========

    def _load_cross_experiment_prompts(self, dataset_name: str, max_prompts: int = 3) -> list:
        """Загрузить лучшие промты из предыдущих экспериментов того же датасета."""
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
        if not os.path.exists(results_dir):
            return []

        best_prompts = []

        # Scan all result directories for matching dataset
        for summary_path in glob_module.glob(os.path.join(results_dir, '**', 'summary.json'), recursive=True):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)

                # Check if this is the same dataset
                if dataset_name.lower() not in summary_path.lower():
                    continue

                best_text = summary.get('best_prompt', summary.get('best_prompt_text', ''))
                best_fitness = summary.get('best_fitness', 0.0)

                if best_text and len(best_text) > 10 and best_fitness > 0.1:
                    best_prompts.append({
                        'text': best_text,
                        'fitness': best_fitness,
                        'source': summary_path
                    })
            except Exception:
                continue

        # Sort by fitness, return top-K
        best_prompts.sort(key=lambda x: x['fitness'], reverse=True)

        # Deduplicate by text similarity (first 100 chars)
        unique = []
        for p in best_prompts:
            if not any(p['text'][:100] == u['text'][:100] for u in unique):
                unique.append(p)
            if len(unique) >= max_prompts:
                break

        if unique:
            logger.info(f"CROSS-EXP MEMORY: Loaded {len(unique)} prompts from previous {dataset_name} experiments")
            for p in unique:
                logger.info(f"  fitness={p['fitness']:.4f} from {os.path.basename(os.path.dirname(p['source']))}")

        return unique

    # ========== Initialization ==========

    def initialize_population(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        show_progress: bool = True
    ) -> List[Prompt]:
        """
        Diverse seeding с data-aware candidates.

        Стратегии:
        - zero_order промпты с разными температурами (0.5, 0.7, 0.9, 1.1, 1.3)
        - APE-style data-aware промпты (LLM видит реальные input/output)
        - concept brainstorming промпты (разные фреймингы задачи)
        - task-specific fallback templates (гарантированно рабочие)

        Args:
            train_data: Train данные для data-aware seeding
            val_data: Validation данные для оценки
            show_progress: Показывать прогресс

        Returns:
            Список промптов (population)
        """
        population = []
        task_desc = self.operators.get_task_description(self.dataset_name)
        population_size = self.config['population_size']

        logger.info(f"Diverse seeding — initializing {population_size} prompts...")

        # Cross-experiment memory — load best prompts from previous experiments
        # Disable via config for fair baseline comparison (cross_experiment_memory=False)
        if self.cross_experiment_memory:
            cross_exp_prompts = self._load_cross_experiment_prompts(self.dataset_name, max_prompts=3)
        else:
            logger.info("Cross-Experiment Memory DISABLED for fair baseline comparison")
            cross_exp_prompts = []

        # concept_brainstorm убран (0% success на ВСЕХ датасетах/моделях).
        # Слоты перенесены в data-aware APE-style seeding (5 слотов вместо 2).
        # Replace fallback templates with cross-experiment prompts if available.
        # BUG FIX: cross_exp_prompts counts toward population budget,
        # otherwise they get truncated by population = population[:population_size] at the end.
        num_cross_exp = len(cross_exp_prompts)
        num_fallback = max(0, 2 - num_cross_exp)
        num_data_aware = 5  # было 2, concept_brainstorm слоты перенесены сюда
        num_zero_order = max(1, population_size - num_fallback - num_data_aware - num_cross_exp)

        # --- Стратегия 1 + 2: Parallel generation (zero_order + data_aware) ---
        temperatures = [0.5, 0.7, 0.9, 1.1, 1.3]
        data_aware_temps = [0.8, 1.0, 1.2, 0.9, 1.1]

        def _gen_zero_order(i):
            temp = temperatures[i % len(temperatures)]
            prompt = self.operators.zero_order_generation(
                task_desc, temperature=temp, dataset_name=self.dataset_name
            )
            prompt.id = f"init_zo_{i}"
            prompt.generation = 0
            return ('zo', i, temp, prompt)

        def _gen_data_aware(i):
            temp = data_aware_temps[i % len(data_aware_temps)]
            data_prompt = self._generate_data_aware_prompt(
                train_data, self.dataset_name, task_desc, temperature=temp
            )
            data_prompt.id = f"init_data_aware_{i}"
            data_prompt.generation = 0
            return ('da', i, temp, data_prompt)

        max_workers = min(32, num_zero_order + num_data_aware)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_zero_order):
                futures.append(executor.submit(_gen_zero_order, i))
            for i in range(num_data_aware):
                futures.append(executor.submit(_gen_data_aware, i))

            # Collect results in order
            zo_results = {}
            da_results = {}
            for future in as_completed(futures):
                kind, idx, temp, prompt = future.result()
                if kind == 'zo':
                    zo_results[idx] = (temp, prompt)
                else:
                    da_results[idx] = (temp, prompt)

        # Add in order
        for i in range(num_zero_order):
            temp, prompt = zo_results[i]
            population.append(prompt)
            if show_progress:
                print(f"  [{i+1}/{population_size}] zero_order (T={temp}): {prompt.text[:60].encode('ascii', 'replace').decode()}...")

        # concept_brainstorm убран (0% success), все 5 слотов — data-aware
        for i in range(num_data_aware):
            temp, prompt = da_results[i]
            population.append(prompt)
            if show_progress:
                print(f"  [{num_zero_order + i + 1}/{population_size}] data_aware (T={temp}): {prompt.text[:60].encode('ascii', 'replace').decode()}...")

        # --- Стратегия 3: task-specific fallback templates ---
        from rider.core.operators import EvolutionaryOperators
        fallback_base = EvolutionaryOperators.TASK_FALLBACK_PROMPTS.get(
            self.dataset_name, f"Solve this task: {task_desc}"
        )
        # XSum gets additional "completion" framing fallback.
        # v11 analysis: PB won XSum with "Complete the news summary..." (18 words, 0.496).
        # RIDER was stuck in "Summarize/Distill" cluster. Adding completion framing to seeding.
        if self.dataset_name == 'XSum':
            fallback_prompts = [
                fallback_base,
                "Complete the news summary by writing the opening lead sentence that the rest of the article follows from."
            ]
        else:
            fallback_prompts = [
                fallback_base,
                f"You are an expert. {fallback_base} Be precise in your output."
            ]
        for i, fb_text in enumerate(fallback_prompts[:num_fallback]):
            fb_prompt = Prompt(
                text=fb_text,
                generation=0,
                mutation_type="fallback_template"
            )
            fb_prompt.id = f"init_fallback_{i}"
            population.append(fb_prompt)
            if show_progress:
                idx = num_zero_order + num_data_aware + i + 1
                print(f"  [{idx}/{population_size}] fallback: {fb_text[:60]}...")

        # Add cross-experiment prompts (replace fallback slots)
        for i, xp in enumerate(cross_exp_prompts):
            xp_prompt = Prompt(
                text=xp['text'],
                generation=0,
                mutation_type="cross_experiment"
            )
            xp_prompt.id = f"init_cross_exp_{i}"
            population.append(xp_prompt)
            if show_progress:
                idx = len(population)
                print(f"  [{idx}/{population_size}] cross_exp (prev_fitness={xp['fitness']:.4f}): {xp['text'][:60]}...")

        # CHIMERA ENGINE: seed ALL prompts with few-shot examples for generative tasks
        # FIX 4: Was only 3 of 12 — now ALL get few-shot context from the start.
        # Each prompt gets different random examples for diversity.
        if self._is_generative_task(self.dataset_name, self.evaluator) and train_data:
            seed_count = len(population)
            for i in range(seed_count):
                if not population[i].few_shot_examples:
                    population[i].few_shot_examples = random.sample(
                        train_data, min(2, len(train_data))
                    )
            logger.info(f"CHIMERA: seeded ALL {seed_count} prompts with few-shot examples")

        # Обрезаем до population_size
        population = population[:population_size]

        # Оцениваем начальную популяцию на VAL
        if show_progress:
            print(f"Evaluating {len(population)} initial prompts on VAL...")

        # Parallel evaluation of initial population
        eval_results = {}  # idx -> eval_details or None

        def _eval_init_prompt(idx_prompt):
            idx, p = idx_prompt
            try:
                if self.config.get('log_detailed_evaluations', True):
                    ed = self.evaluator.evaluate_with_details(p, self.dataset_name, val_data)
                    pm = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    p.fitness = ed['metrics'][pm]
                    return idx, ed
                else:
                    p.fitness = self.evaluator.evaluate_prompt(
                        p, self.dataset_name, val_data, use_cache=True
                    )
                    return idx, None
            except Exception as e:
                logger.error(f"Init eval failed for prompt {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = min(4, len(population))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, ed in executor.map(_eval_init_prompt, enumerate(population)):
                eval_results[idx] = ed

        # Sequential logging (history not thread-safe)
        for idx, prompt in enumerate(population):
            ed = eval_results.get(idx)
            if ed and self.config.get('log_detailed_evaluations', True):
                self.history.log_detailed_evaluation(
                    prompt_id=prompt.id,
                    generation=0,
                    dataset_name=self.dataset_name,
                    evaluation_details=ed
                )

            self.history.log_evolution_step(
                generation=0,
                operator_used=prompt.metadata.get('mutation_type', 'zero_order'),
                parent_ids=[],
                parent_fitnesses=[],
                offspring=prompt,
                temperature=self.temperature,
                top_p=0.95,
                diversity_score=0.0,
                accepted=True,
                rejection_reason=None,
                metadata={}
            )

        logger.info(
            f"RIDER diverse seeding complete: {len(population)} prompts, "
            f"avg_fitness={np.mean([p.fitness for p in population]):.4f}, "
            f"best_fitness={max(p.fitness for p in population):.4f}"
        )

        return population

    # ========== Selection ==========

    def select_parents(self) -> Tuple[Prompt, Prompt]:
        """
        Tournament selection для выбора родителей.

        Returns:
            Tuple из двух родительских промптов
        """
        tournament_size = self.config.get('tournament_size', 3)

        parent1 = max(
            random.sample(self.population, min(tournament_size, len(self.population))),
            key=lambda p: p.fitness
        )
        parent2 = max(
            random.sample(self.population, min(tournament_size, len(self.population))),
            key=lambda p: p.fitness
        )

        return parent1, parent2

    # ========== Stagnation Escape ==========

    def _detect_and_respond_to_stagnation(self, generation: int, best_fitness: float) -> None:
        """
        Детектирует стагнацию и адаптирует температуру / top_p.
        При глубокой стагнации (>=4 gen) выполняет soft restart популяции.
        Diversity injection: при stagnation_count >= 1 AND diversity < 0.30.
        """
        self.stagnation_fitness_history.append(best_fitness)

        # Окно проверки уменьшено с 3 до 2 поколений.
        # При 8 gen стагнация детектировалась слишком поздно (Gen 5 mild, Gen 7 deep).
        if len(self.stagnation_fitness_history) < 2:
            return

        # Проверяем improvement за последние 2 поколения
        recent = self.stagnation_fitness_history[-2:]
        improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 0.001)

        prev_stagnation = self.stagnation_count  # запомним для fix temperature recovery

        if improvement < 0.01:  # < 1% improvement
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            # Температура восстанавливается только если ДО этого не было стагнации.
            # Иначе единичный noise spike сбрасывал stagnation_count и уменьшал T — баг XSum.
            if prev_stagnation == 0:
                self.current_temperature = max(
                    self.base_temperature,
                    self.current_temperature * 0.9
                )

        # Сигнализируем UCB о стагнации
        # порог снижен с 3 до 2 — быстрее реагируем
        if hasattr(self.ucb_selector, 'set_stagnation'):
            self.ucb_selector.set_stagnation(self.stagnation_count >= 2)

        # Mild stagnation (2+ gen, было 3+): повышаем температуру
        if self.stagnation_count >= 2:
            # T cap 1.3. multiplier 1.3 (было 1.2) — быстрее escape
            self.current_temperature = min(1.3, self.current_temperature * 1.3)
            self.current_top_p = min(1.0, self.current_top_p + 0.03)
            logger.info(
                f"Gen {generation}: STAGNATION ({self.stagnation_count} gens). "
                f"T={self.current_temperature:.2f}, top_p={self.current_top_p:.2f}"
            )

        # Diversity injection — раньше и с более высоким порогом.
        # v12 analysis: CommonGen haiku diversity collapsed 0.445→0.174 by Gen 4 (stagnation=1),
        # but injection required stagnation >= 2. By then diversity was already 0.165.
        # Changed: stagnation >= 1 + diversity < 0.30 (was >= 2 + < 0.25).
        if self.stagnation_count >= 1 and hasattr(self, 'diversity_manager'):
            diversity = self.diversity_manager.compute_diversity_score(self.population)
            if diversity < 0.30:
                self._diversity_injection(generation)

        # Deep stagnation (4+ gen, было 5+): soft restart раньше,
        # чтобы успеть до последнего поколения (Gen 7 при 8 gen)
        if self.stagnation_count >= 4:
            # Не увеличиваем T дальше 1.3, вместо этого soft restart
            logger.info(
                f"Gen {generation}: DEEP STAGNATION ({self.stagnation_count} gens). "
                f"T={self.current_temperature:.2f} → triggering soft restart"
            )
            self._soft_restart(generation)

    def _soft_restart(self, generation: int) -> None:
        """
        Soft restart: сохраняем top-50% элит, заменяем остальных свежими промптами.
        генерируем при умеренной T (0.7-1.0) вместо escalated T,
        и сразу оцениваем на val_data чтобы промпты не входили с fitness=0.0.
        """
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        keep_count = max(3, len(sorted_pop) // 2)
        task_desc = self.operators.get_task_description(self.dataset_name)

        fresh_prompts = []
        for _ in range(len(sorted_pop) - keep_count):
            fresh = self.operators.zero_order_generation(
                task_desc, dataset_name=self.dataset_name,
                temperature=random.uniform(0.7, 1.0)  # V10-4: умеренная T
            )
            fresh.generation = generation
            fresh_prompts.append(fresh)

        # V10-4: Сразу оцениваем свежие промпты чтобы не входили с fitness=0.0
        if hasattr(self, '_val_data') and self._val_data:
            for p in fresh_prompts:
                try:
                    p.fitness = self.evaluator.evaluate_prompt(
                        p, self.dataset_name, self._val_data, use_cache=True
                    )
                except Exception as e:
                    logger.warning(f"Soft restart eval failed: {e}")
                    p.fitness = 0.0

        self.population = sorted_pop[:keep_count] + fresh_prompts
        self._restart_count += 1
        self.stagnation_count = 2  # Не сбрасываем полностью — позволяем early stopping сработать после ещё одного цикла
        # Отдельный restart counter нужен потому что stagnation_count после soft restart
        # специально откатывается до 2 и сам по себе больше не может надёжно дойти до early stop.

        logger.info(
            f"Gen {generation}: SOFT RESTART — kept top {keep_count}, "
            f"injected {len(fresh_prompts)} fresh prompts"
        )

    def _diversity_injection(self, generation: int) -> None:
        """
        Diversity injection — заменяет 25% худших промптов свежими.
        Вызывается при стагнации + diversity < 0.30 (но до deep stagnation).
        Мягче soft restart: сохраняет 75% популяции, включая всю элиту.
        умеренная T + immediate eval.
        """
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        replace_count = max(2, len(sorted_pop) // 4)  # 25% популяции
        keep_count = len(sorted_pop) - replace_count
        task_desc = self.operators.get_task_description(self.dataset_name)

        fresh_prompts = []
        for _ in range(replace_count):
            fresh = self.operators.zero_order_generation(
                task_desc, dataset_name=self.dataset_name,
                temperature=random.uniform(0.7, 1.0)  # V10-4: умеренная T (было 0.8-1.3)
            )
            fresh.generation = generation
            fresh_prompts.append(fresh)

        # V10-4: Сразу оцениваем свежие промпты
        if hasattr(self, '_val_data') and self._val_data:
            for p in fresh_prompts:
                try:
                    p.fitness = self.evaluator.evaluate_prompt(
                        p, self.dataset_name, self._val_data, use_cache=True
                    )
                except Exception as e:
                    logger.warning(f"Diversity injection eval failed: {e}")
                    p.fitness = 0.0

        self.population = sorted_pop[:keep_count] + fresh_prompts

        logger.info(
            f"Gen {generation}: DIVERSITY INJECTION — kept top {keep_count}, "
            f"replaced {replace_count} worst with fresh prompts"
        )

    def _nexus_classify_examples(self, val_data: List[Dict]) -> List[int]:
        """NEXUS PROTOCOL v2 — continuous difficulty scoring.

        Instead of SOLVED/CRITICAL/IMPOSSIBLE trichotomy (broken for BERTScore tasks),
        uses variance of fitness across population as discriminative power.
        High variance = high discriminative power = valuable for evaluation.

        Returns indices of most discriminative examples (top 33%, min 5),
        preserving variance-descending order for PRISM.
        """
        if not self.population or not any(p.fitness > 0 for p in self.population):
            return list(range(len(val_data)))  # No data yet, use all

        # Evaluate top-6 prompts on each example individually.
        # Parallelize across examples; keep per-example prompt loop sequential so
        # each variance value is deterministic and easy to reason about.
        top_prompts = sorted(self.population, key=lambda p: p.fitness, reverse=True)[:6]

        def _score_example(i_example):
            i, example = i_example
            scores = []
            for prompt in top_prompts:
                try:
                    score = self.evaluator.evaluate_prompt(
                        prompt=prompt, dataset_name=self.dataset_name, data=[example],
                        use_cache=False  # NEXUS: no cache to avoid collision + fitness corruption
                    )
                    scores.append(score)
                except Exception:
                    scores.append(0.0)

            # Variance = discriminative power
            # BUG FIX: avoid UnboundLocalError when len(scores) < 2.
            mean_score = (sum(scores) / len(scores)) if scores else 0.0
            if len(scores) >= 2:
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            else:
                variance = 0.0

            return (i, variance, mean_score)

        nexus_workers = min(4, len(val_data))
        if nexus_workers > 1:
            with ThreadPoolExecutor(max_workers=nexus_workers) as executor:
                example_variance = list(executor.map(_score_example, enumerate(val_data)))
        else:
            example_variance = [_score_example(item) for item in enumerate(val_data)]

        # Sort by variance (highest = most discriminative)
        example_variance.sort(key=lambda x: x[1], reverse=True)

        # Take top 33% by variance (min 5) — reduced from 50% for API savings
        num_critical = max(5, len(val_data) // 3)
        critical_indices = [idx for idx, var, mean in example_variance[:num_critical]]

        logger.info(
            f"NEXUS v2: top-{num_critical} discriminative examples "
            f"(max_var={example_variance[0][1]:.4f}, min_var={example_variance[-1][1]:.4f})"
        )

        return critical_indices

    @staticmethod
    def _racing_should_eliminate(scores_candidate: list, scores_leader: list, min_samples: int = 6) -> bool:
        """F-Race: eliminate candidate if statistically worse than leader.

        Uses Wilcoxon signed-rank test on paired per-example scores.
        If we can reject H0 (candidate >= leader) at p < 0.05 with at least
        min_samples shared examples, the candidate is eliminated early.

        Ref: Birattari et al. 2002 "A Racing Algorithm for Configuring Metaheuristics" (GECCO).
        """
        if len(scores_candidate) < min_samples or len(scores_leader) < min_samples:
            return False
        try:
            from scipy.stats import wilcoxon
            n = min(len(scores_candidate), len(scores_leader))
            diffs = [scores_leader[i] - scores_candidate[i] for i in range(n)]
            if all(d == 0 for d in diffs):
                return False
            _, p_value = wilcoxon(diffs, alternative='greater')
            return p_value < 0.05
        except Exception:
            return False

    def _normalize_token_metric_text(self, text) -> str:
        """Normalize text for token-overlap metrics without depending on test doubles."""
        metrics_evaluator = getattr(self.evaluator, 'metrics_evaluator', None)
        normalize_answer = getattr(metrics_evaluator, 'normalize_answer', None)
        if callable(normalize_answer):
            try:
                return normalize_answer(str(text))
            except Exception:
                pass

        normalized = str(text).lower().strip()
        normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in normalized)
        normalized = ' '.join(tok for tok in normalized.split() if tok not in {'a', 'an', 'the'})
        return normalized

    def _single_example_f1_score(self, prediction, truth) -> float:
        """Compute token-level F1 for QA-style metrics on one prediction/reference pair."""
        pred_normalized = self._normalize_token_metric_text(prediction)

        if isinstance(truth, dict):
            truth_answers = truth.get('answers', [])
            is_impossible = truth.get('is_impossible', len(truth_answers) == 0)
            if is_impossible:
                return 1.0 if (
                    pred_normalized == '' or
                    'impossible' in pred_normalized or
                    'cannot' in pred_normalized or
                    'no answer' in pred_normalized
                ) else 0.0

            candidate_truths = truth_answers or ['']
        else:
            candidate_truths = [truth]

        max_f1 = 0.0
        pred_tokens = pred_normalized.split()
        for candidate_truth in candidate_truths:
            truth_normalized = self._normalize_token_metric_text(candidate_truth)
            truth_tokens = truth_normalized.split()

            if pred_normalized == truth_normalized:
                return 1.0
            if not pred_tokens or not truth_tokens:
                continue

            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            token_f1 = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
            max_f1 = max(max_f1, token_f1)

        return max_f1

    def _single_example_metric_score(self, metric: str, prediction, truth, data_item=None) -> float:
        """Compute one-example metric scores without issuing extra LLM calls."""
        metrics_evaluator = getattr(self.evaluator, 'metrics_evaluator', None)
        evaluate = getattr(metrics_evaluator, 'evaluate', None)
        if callable(evaluate):
            try:
                metrics = evaluate(
                    self.dataset_name,
                    [prediction],
                    [truth],
                    data=[data_item] if data_item is not None else None
                )
                if isinstance(metrics, dict):
                    metric_value = metrics.get(metric)
                    if isinstance(metric_value, (int, float, np.floating)):
                        return float(metric_value)
            except Exception:
                pass

        if metric == 'f1':
            return float(self._single_example_f1_score(prediction, truth))
        if metric == 'f1_macro':
            return 1.0 if str(prediction).strip().lower() == str(truth).strip().lower() else 0.0
        if metric == 'exact_match':
            return 1.0 if str(prediction).strip().lower() == str(truth).strip().lower() else 0.0
        return 0.0

    def _extract_per_example_scores(self, metric: str, eval_details: Dict, batch_data: List[Dict]) -> List[float]:
        """Turn a detailed batch evaluation into per-example scores for PRISM/F-Race."""
        predictions = eval_details.get('predictions', [])
        truths = eval_details.get('ground_truth', [])
        limit = min(len(batch_data), len(predictions), len(truths))

        per_example_scores = [
            self._single_example_metric_score(metric, predictions[i], truths[i], data_item=batch_data[i])
            for i in range(limit)
        ]
        if limit < len(batch_data):
            per_example_scores.extend([0.0] * (len(batch_data) - limit))
        return per_example_scores

    def _get_phase_config(self, generation: int) -> Dict:
        """PHASE REACTOR — temperature and offspring modulation.

        All operators remain available. Historical phase boosts remain in
        _PHASE_CONFIG for compatibility/analysis, but are not applied to the
        Thompson selector because empirical runs favored unboosted Thompson
        Sampling.

        Returns dict with 'name', 'boosts', 'temperature', 'offspring_mult'.
        Should be called at the START of evolve_generation() to determine
        temperature multiplier and offspring count.
        """
        config = self._PHASE_CONFIG.get(generation, self._PHASE_CONFIG[7])  # Default to last phase

        phase_temp = self.base_temperature * config['temperature_mult']

        phase = {
            'name': config['name'],
            'boosts': config.get('boosts', {}),
            'temperature': phase_temp,
            'offspring_mult': config['offspring_mult'],
        }

        # BUG FIX: PHASE REACTOR temperature must be assigned to self.current_temperature.
        # Without this, temperature stays at init value during CRYSTALLIZATION phase (Gen 5-7),
        # producing garbage offspring at too-high temperature.
        self.current_temperature = phase_temp

        logger.info(
            f"PHASE REACTOR: {phase['name']} (Gen {generation}) — "
            f"T={phase_temp:.2f}, "
            f"offspring={phase['offspring_mult']}x"
        )

        return phase

    def _get_error_type_operator_bias(self, generation: int) -> Optional[Dict[str, float]]:
        """
        Instance-Aware Operator Selection.
        Analyzes error patterns and returns operator bias based on which operators
        historically produce prompts that fix similar error types.

        GEPA uses instance-wise Pareto for prompts. We do instance-wise credit for OPERATORS.
        This is something GEPA cannot do with its single operator.
        """
        if not hasattr(self, '_error_operator_history'):
            self._error_operator_history = []  # List of (error_type, operator, improvement)

        if not self.current_errors or generation < 2:
            return None

        # Classify current errors into types
        error_types = set()
        for err in self.current_errors[:10]:
            pred = str(err.get('prediction', '')).lower()
            truth = str(err.get('ground_truth', '')).lower()

            # Simple error classification
            if not pred.strip() or pred.strip() in ('', 'none', 'n/a'):
                error_types.add('empty_output')
            elif len(pred) > len(truth) * 3:
                error_types.add('too_verbose')
            elif len(pred) < len(truth) * 0.3 and len(truth) > 5:
                error_types.add('too_short')
            else:
                error_types.add('wrong_content')

        if not error_types:
            return None

        # Look up which operators historically fixed these error types
        operator_scores = defaultdict(list)
        for hist_error_type, hist_op, hist_improvement in self._error_operator_history:
            if hist_error_type in error_types and hist_improvement > 0:
                operator_scores[hist_op].append(hist_improvement)

        if not operator_scores:
            return None

        # Convert to bias dict: operators that fix our current error types get bonus
        bias = {}
        for op, improvements in operator_scores.items():
            bias[op] = min(0.3, np.mean(improvements) * 2)  # Cap at 0.3 bonus

        logger.debug(
            f"Gen {generation}: Instance-aware bias for error types {error_types}: {bias}"
        )
        return bias

    # ========== CHIMERA ENGINE: Few-Shot Evolution ==========

    def _chimera_swap_examples(self, parent, train_data):
        """CHIMERA: swap one few-shot example with random train example."""
        offspring = Prompt(
            text=parent.text,
            few_shot_examples=copy.deepcopy(parent.few_shot_examples or [])
        )
        if not offspring.few_shot_examples:
            offspring.few_shot_examples = random.sample(train_data, min(2, len(train_data)))
        else:
            idx = random.randint(0, len(offspring.few_shot_examples) - 1)
            offspring.few_shot_examples[idx] = random.choice(train_data)
        offspring.parent_ids = [parent.id]
        offspring.mutation_type = 'chimera_swap'
        return offspring

    def _chimera_crossover(self, parent1, parent2):
        """CHIMERA: instruction from parent1, examples from parent2."""
        return Prompt(
            text=parent1.text,
            few_shot_examples=copy.deepcopy(parent2.few_shot_examples or []),
            parent_ids=[parent1.id, parent2.id],
            mutation_type='chimera_crossover'
        )

    # ========== Operator Application ==========

    def apply_operator(
        self,
        operator_name: str,
        generation: int,
        eval_samples: Optional[List[Dict]] = None
    ) -> Prompt:
        """
        Применяет выбранный оператор.

        Args:
            operator_name: Название оператора
            generation: Номер поколения
            eval_samples: Примеры для evaluation (опционально)

        Returns:
            Offspring промпт
        """
        task_desc = self.operators.get_task_description(self.dataset_name)

        # OPERATOR FORGE — inject operator self-memory as few-shot context
        forge_context = self.operator_forge.get_context(operator_name)
        if forge_context:
            task_desc = task_desc + forge_context

        # GENESIS — inject ancestral lessons
        genesis_context = self.genesis.get_context(operator_name)
        if genesis_context:
            task_desc = task_desc + genesis_context

        # Task-adaptive filtering
        if operator_name in self._UNIVERSALLY_EXCLUDED:
            logger.debug(
                f"Task-adaptive: {operator_name} excluded universally, using zero_order"
            )
            operator_name = 'zero_order'
        elif self._is_generative_task(self.dataset_name, self.evaluator) and operator_name in self._GENERATIVE_EXCLUDED:
            logger.debug(
                f"Task-adaptive: {operator_name} excluded for {self.dataset_name}, "
                "using zero_order instead"
            )
            operator_name = 'zero_order'

        # UCB reward vs median (было vs best).
        # EDA создаёт промпт из top-K → он всегда МЕЖДУ ними по quality →
        # fitness < best → "неудача" → Thompson наказывает EDA. С медианой EDA получает
        # заслуженные положительные rewards.
        median_fitness = float(np.median([p.fitness for p in self.population]))

        # Применяем оператор
        if operator_name == 'ga_mutation':
            parent = self.select_parents()[0]
            offspring = self.operators.ga_mutation(parent, task_desc)

        elif operator_name == 'de_mutation':
            base = random.choice(self.population)
            donors = random.sample(self.population, min(2, len(self.population)))
            best = max(self.population, key=lambda p: p.fitness)
            offspring = self.operators.de_mutation(
                base, donors[0], donors[1] if len(donors) > 1 else donors[0],
                best, task_desc
            )

        elif operator_name == 'zero_order':
            offspring = self.operators.zero_order_generation(
                task_desc, dataset_name=self.dataset_name
            )

        elif operator_name == 'eda_mutation':
            offspring = self.operators.eda_mutation(self.population, task_desc)

        elif operator_name == 'eda_rank_index':
            offspring = self.operators.eda_rank_index(self.population, task_desc)

        elif operator_name == 'reflection_crossover':
            parent1, parent2 = self.select_parents()
            # Добавляем memory context для обогащённой рефлексии
            enriched_desc = task_desc
            ctx = self.long_term_memory.get_context()
            if ctx.get('success'):
                enriched_desc += "\nSuccessful patterns: " + "; ".join(ctx['success'][:2])
            if ctx.get('failure'):
                enriched_desc += "\nAvoid patterns: " + "; ".join(ctx['failure'][:2])
            offspring = self.operators.reflection_crossover(parent1, parent2, enriched_desc)

        elif operator_name == 'opro_trajectory_mutation':
            # OPRO-style — LLM видит топ-20 промптов с scores
            offspring = self.operators.opro_trajectory_mutation(
                self.population, task_desc, dataset_name=self.dataset_name
            )

        elif operator_name == 'contrastive_error_decomposition':
            # Reverted to simple CED (without trace/memory).
            # v11 analysis: Trace-Enhanced CED DEGRADED performance.
            # CommonGen haiku CED: v10=70.6% success → v11=37% success.
            # Extra context (failed_attempts + memory) confused LLM.
            elite = max(self.population, key=lambda p: p.fitness)
            offspring = self.operators.contrastive_error_decomposition(
                elite, self.current_errors, self.population, task_desc,
                dataset_name=self.dataset_name
            )

        elif operator_name == 'semantic_paraphrase':
            # PhaseEvo-style — семантический парафраз
            parent = self.select_parents()[0]
            offspring = self.operators.semantic_paraphrase(
                parent, task_desc, dataset_name=self.dataset_name
            )

        else:
            logger.warning(f"Unknown operator: {operator_name}, using zero_order")
            offspring = self.operators.zero_order_generation(
                task_desc, dataset_name=self.dataset_name
            )

        # CHIMERA: inherit few_shot_examples from parent if offspring doesn't have its own
        if not offspring.few_shot_examples:
            # Find the parent that was used for this operator
            parent_with_examples = None
            if offspring.parent_ids:
                for p in self.population:
                    if p.id in offspring.parent_ids and p.few_shot_examples:
                        parent_with_examples = p
                        break
            if parent_with_examples:
                offspring.few_shot_examples = copy.deepcopy(parent_with_examples.few_shot_examples)

        # Сохраняем metadata для UCB update
        offspring.metadata['used_operator'] = operator_name
        offspring.metadata['parent_fitness'] = median_fitness  # vs median, не vs best
        offspring.generation = generation

        return offspring

    # ========== Pareto Selection ==========

    def pareto_selection(
        self,
        candidates: List[Prompt],
        k: int
    ) -> List[Prompt]:
        """
        Pareto-отбор на основе fitness + diversity.

        Args:
            candidates: Кандидаты для отбора
            k: Количество для выбора

        Returns:
            k отобранных промптов
        """
        if len(candidates) <= k:
            return candidates

        # Вычисляем embeddings
        self.diversity_manager.compute_embeddings(candidates)

        pareto_front = []
        remaining = candidates.copy()

        while remaining and len(pareto_front) < k:
            non_dominated = []

            for candidate in remaining:
                is_dominated = False

                # Вычисляем diversity score
                if pareto_front:
                    candidate_div = min(
                        1 - np.dot(
                            candidate.diversity_features,
                            p.diversity_features
                        ) / (
                            np.linalg.norm(candidate.diversity_features) *
                            np.linalg.norm(p.diversity_features) + 1e-8
                        )
                        for p in pareto_front
                    )
                else:
                    candidate_div = 1.0

                # Проверка доминирования
                for other in remaining:
                    if other.id == candidate.id:
                        continue

                    if pareto_front:
                        other_div = min(
                            1 - np.dot(
                                other.diversity_features,
                                p.diversity_features
                            ) / (
                                np.linalg.norm(other.diversity_features) *
                                np.linalg.norm(p.diversity_features) + 1e-8
                            )
                            for p in pareto_front
                        )
                    else:
                        other_div = 1.0

                    # other доминирует candidate
                    if (other.fitness >= candidate.fitness and
                        other_div >= candidate_div and
                        (other.fitness > candidate.fitness or other_div > candidate_div)):
                        is_dominated = True
                        break

                if not is_dominated:
                    non_dominated.append(candidate)

            # Добавляем недоминируемые
            if non_dominated:
                if len(pareto_front) + len(non_dominated) > k:
                    needed = k - len(pareto_front)
                    # Сортируем по diversity
                    if pareto_front:
                        non_dominated.sort(
                            key=lambda p: min(
                                1 - np.dot(p.diversity_features, pf.diversity_features) /
                                (np.linalg.norm(p.diversity_features) * np.linalg.norm(pf.diversity_features) + 1e-8)
                                for pf in pareto_front
                            ),
                            reverse=True
                        )
                    else:
                        non_dominated.sort(key=lambda p: p.fitness, reverse=True)

                    pareto_front.extend(non_dominated[:needed])
                else:
                    pareto_front.extend(non_dominated)

                for p in non_dominated:
                    if p in remaining:
                        remaining.remove(p)
            else:
                # Fallback
                remaining.sort(key=lambda p: p.fitness, reverse=True)
                needed = k - len(pareto_front)
                pareto_front.extend(remaining[:needed])
                break

        return pareto_front

    # ========== Evolution ==========

    def evolve_generation(
        self,
        generation: int,
        val_data: Optional[List[Dict]] = None,
        show_progress: bool = False
    ) -> None:
        """
        Evaluate-first + (mu+lambda) retention.

        Ключевое отличие: все offspring оцениваются ДО решения об их включении в популяцию.
        - (mu+lambda): объединяем parents + offspring, сортируем по fitness, оставляем top-N
        - Diversity = tiebreaker (не gate): отклоняет только near-identical дубликаты
        - UCB reward vs median (не vs best) — EDA получает заслуженные positive rewards

        Args:
            generation: Номер поколения
            val_data: Validation данные для оценки offspring
            show_progress: Показывать прогресс
        """
        population_size = self.config['population_size']

        # ===== PHASE REACTOR — get phase config =====
        phase = self._get_phase_config(generation)

        # BUG FIX: propagate PHASE REACTOR temperature to operators.
        # _get_phase_config() sets self.current_temperature, but operators read their
        # own self.temperature — without this assignment, CRYSTALLIZATION (Gen 5-7)
        # never actually lowers operator temperature, so generation stays hot.
        if hasattr(self, 'operators'):
            self.operators.temperature = self.current_temperature

        # Phase boosts DISABLED — Thompson Sampling adapts better on its own.
        # Boosts retained in _PHASE_CONFIG for backward compat (tests), but not applied.
        # if hasattr(self, 'ucb_selector') and hasattr(self.ucb_selector, 'set_phase_boosts'):
        #     self.ucb_selector.set_phase_boosts(phase.get('boosts', {}))

        # ===== Step 1: Elite Selection (Change C: pure fitness) =====
        elite_size = self.config['elite_size']
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        elites = sorted_pop[:elite_size]

        # Сохраняем элитную историю
        if elites:
            self.elite_history.append(elites[0])
            self.elite_history = self.elite_history[-10:]

        # ===== Long-term Memory Update =====
        memory_update_interval = self.config.get('memory_update_interval', 3)
        if generation > 0 and generation % memory_update_interval == 0:
            self.long_term_memory.update(self.population, generation)

        # ===== Step 2: Generate offspring =====

        # P0-2: Adaptive offspring count — PHASE REACTOR drives non-stagnation cases.
        # Stagnation overrides phase config for maximum exploration.
        if self.stagnation_count >= 3:
            offspring_multiplier = 4.0  # Deep stagnation → maximum exploration
        elif self.stagnation_count >= 1:
            offspring_multiplier = 3.0  # Mild stagnation → more exploration
        else:
            # Use PHASE REACTOR offspring_mult (decreases over evolution)
            offspring_multiplier = phase['offspring_mult']
        num_offspring = max(population_size, int(population_size * offspring_multiplier))

        # Store for generation summary logging
        self._last_offspring_multiplier = offspring_multiplier
        self._last_num_offspring = num_offspring

        logger.info(
            f"Gen {generation}: offspring_multiplier={offspring_multiplier}x → "
            f"{num_offspring} offspring (stagnation_count={self.stagnation_count})"
        )

        # Bayesian Automatic Operator Elimination.
        # После Gen 1 используем Bayesian posterior test вместо хардкода.
        # автодетект по метрике вместо хардкода имён датасетов.
        # Ref: Even-Dar et al. 2006 "Action Elimination" (JMLR).
        excluded_ops = list(self._UNIVERSALLY_EXCLUDED)
        bayesian_eliminated = []
        if generation <= 1:
            # Fallback: автодетект для первых поколений (недостаточно данных для Bayesian)
            if self._is_generative_task(self.dataset_name, self.evaluator):
                excluded_ops.extend(list(self._GENERATIVE_EXCLUDED))
        else:
            # Reverted to P>0.95 (v11 used 0.85 which was too aggressive).
            # v11 analysis: P>0.85 eliminated working operators:
            #   - CED (50% success on SQuAD_2 haiku) eliminated Gen 6
            #   - zero_order (50% success on CommonGen haiku) eliminated Gen 7
            #   - semantic_paraphrase (33% success on SQuAD_2 gemini) eliminated Gen 6
            # Более агрессивная элиминация.
            # v10 (0.607) элиминировал 5 операторов к Gen 7 → концентрация бюджета на CED.
            # v13 (0.578) элиминировал 0 → бюджет размазан. Причина: confidence=0.95 слишком
            # консервативно + success_rate protection 30% (теперь 15%).
            bayesian_eliminated = self.ucb_selector.get_eliminated_operators(
                min_trials=3, confidence=0.85
            )
            # Auto-eliminate operators with consistently negative improvement.
            # If avg_improvement < -0.10 after 20+ uses → force eliminate.
            # (Raised thresholds from v11: was -0.05/15 which was too aggressive)
            stats = self.ucb_selector.get_statistics()
            for op_name, op_stats in stats.items():
                if op_name in excluded_ops or op_name in bayesian_eliminated:
                    continue
                if op_stats.get('count', 0) >= 20 and op_stats.get('avg_improvement', 0) < -0.10:
                    bayesian_eliminated.append(op_name)
                    logger.info(
                        f"Gen {generation}: AUTO-ELIMINATE {op_name} — "
                        f"avg_improvement={op_stats['avg_improvement']:.4f} < -0.10 "
                        f"after {op_stats['count']} uses"
                    )
            if bayesian_eliminated:
                excluded_ops.extend(bayesian_eliminated)
                logger.info(
                    f"Gen {generation}: Bayesian eliminated operators: {bayesian_eliminated}"
                )
            # автодетект как safety net
            if self._is_generative_task(self.dataset_name, self.evaluator):
                for op in self._GENERATIVE_EXCLUDED:
                    if op not in excluded_ops:
                        excluded_ops.append(op)
            # CED re-enabled for XSum (no longer excluded)

        # Store for generation summary logging
        self._last_eliminated_operators = list(set(excluded_ops))
        self._last_bayesian_eliminated = bayesian_eliminated

        # Instance-aware operator bias
        error_bias = self._get_error_type_operator_bias(generation)

        # Pre-select all operators (fast, no LLM calls — Thompson Sampling)
        selected_operators = []
        for _ in range(num_offspring):
            # 20% chance to use error-biased operator if bias available
            if error_bias and np.random.random() < 0.20:
                biased_ops = [op for op in error_bias if op not in (excluded_ops or [])]
                if biased_ops:
                    weights = np.array([error_bias[op] for op in biased_ops])
                    weights /= weights.sum()
                    selected_operators.append(np.random.choice(biased_ops, p=weights))
                    continue
            selected_operators.append(
                self.ucb_selector.select_operator(
                    generation,
                    self.config['num_generations'],
                    exclude=excluded_ops if excluded_ops else None
                )
            )

        # Parallel offspring generation via ThreadPoolExecutor
        def _generate_one_offspring(operator):
            try:
                return self.apply_operator(operator, generation, val_data)
            except Exception as e:
                logger.error(f"Operator {operator} failed: {e}")
                return None

        all_offspring = []
        short_rejected = 0
        max_workers = min(32, num_offspring)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_generate_one_offspring, op): op
                for op in selected_operators
            }
            for future in as_completed(futures):
                offspring = future.result()
                if offspring is None:
                    continue
                # Change H: minimum word count protection
                # XSum allows shorter prompts (min 10 words).
                # PB won XSum with 18-word prompt. Old min=15 was fine but
                # we lower to 10 to allow concise framings like "Complete the summary..."
                min_words = 10 if self.dataset_name == 'XSum' else 15
                if len(offspring.text.split()) < min_words:
                    short_rejected += 1
                    continue
                all_offspring.append(offspring)

        if short_rejected > 0:
            logger.info(
                f"Gen {generation}: {short_rejected} offspring rejected "
                f"(< 15 words), {len(all_offspring)} kept for evaluation"
            )

        # ===== CHIMERA ENGINE: add few-shot evolved offspring for generative tasks =====
        if self._is_generative_task(self.dataset_name, self.evaluator):
            train_data_ref = getattr(self, '_train_data', [])
            if train_data_ref:
                median_fitness = float(np.median([p.fitness for p in self.population]))
                chimera_count = max(2, population_size // 2)  # FIX 3: more CHIMERA for generative tasks
                for i in range(chimera_count):
                    parent = random.choice(self.population)
                    if i % 2 == 0:
                        ch = self._chimera_swap_examples(parent, train_data_ref)
                    else:
                        p1, p2 = random.sample(self.population, 2)
                        ch = self._chimera_crossover(p1, p2)
                    ch.metadata['used_operator'] = 'chimera'
                    ch.metadata['parent_fitness'] = median_fitness
                    ch.generation = generation
                    all_offspring.append(ch)
                logger.info(f"CHIMERA: added {chimera_count} few-shot offspring")

        # ===== VORTEX — paradigm shift at deep stagnation =====
        if self.stagnation_count >= 3 and hasattr(self, 'operators'):
            try:
                task_desc = self.operators.get_task_description(self.dataset_name)
                vortex_prompt = self.operators.vortex_mutation(
                    population=self.population,
                    task_desc=task_desc,
                    errors=self.current_errors,
                    dataset_name=self.dataset_name,
                    forge_context=self.operator_forge.get_context('vortex')
                )
                if vortex_prompt and len(vortex_prompt.text.split()) >= 10:
                    vortex_prompt.generation = generation
                    vortex_prompt.metadata['used_operator'] = 'vortex'
                    median_fitness = float(np.median([p.fitness for p in self.population]))
                    vortex_prompt.metadata['parent_fitness'] = median_fitness
                    all_offspring.append(vortex_prompt)
                    logger.info(f"VORTEX: paradigm shift generated — {vortex_prompt.text[:80]}...")
            except Exception as e:
                logger.warning(f"VORTEX failed: {e}")

        # ===== Step 3: Semantic Dedup + PRISM (replaces BLITZ) =====
        if val_data and all_offspring:
            # --- Step 3a: Exact text dedup — skip near-identical offspring ---
            evaluated_cache = {}  # dedup_key -> fitness, built from current population
            for p in self.population:
                if p.fitness > 0:
                    cache_key = p.text
                    if p.few_shot_examples:
                        fs_sig = "|".join(
                            str(sorted(ex.items())) for ex in p.few_shot_examples
                        )
                        cache_key = f"{p.text}||{fs_sig}"
                    evaluated_cache[cache_key] = p.fitness

            deduped_offspring = []
            dedup_skipped = []  # Offspring with cached fitness (still participate in merge)
            skipped_count = 0
            for offspring in all_offspring:
                # Dedup key includes few_shot_examples for CHIMERA offspring
                # (same text but different examples = different prompt, must re-evaluate)
                dedup_key = offspring.text
                if offspring.few_shot_examples:
                    fs_sig = "|".join(
                        str(sorted(ex.items())) for ex in offspring.few_shot_examples
                    )
                    dedup_key = f"{offspring.text}||{fs_sig}"

                if dedup_key in evaluated_cache:
                    offspring.fitness = evaluated_cache[dedup_key]
                    offspring.metadata['dedup_skip'] = True
                    dedup_skipped.append(offspring)
                    skipped_count += 1
                    continue

                deduped_offspring.append(offspring)

            logger.info(
                f"Gen {generation}: Semantic dedup: {len(all_offspring)} → "
                f"{len(deduped_offspring)} unique ({skipped_count} duplicates skipped)"
            )

            # --- Step 3b: PRISM — Progressive Refinement via Iterative Successive
            #     halving of Mutations (replaces BLITZ Protocol) ---
            # Ref: Li et al. 2018 "Hyperband" (JMLR), Jamieson & Talwalkar 2016 (AISTATS)
            #
            # Key differences from BLITZ:
            # 1. CUMULATIVE evidence: each round ADDS examples, doesn't replace them
            # 2. Metric-adaptive: exact_match/F1 use 3 examples/round.
            #    BERTScore and f1_macro bypass PRISM and get direct full eval.
            # 3. Mathematical guarantee: P(eliminating true best) = O(1/sqrt(n_examples))
            # 4. NEXUS orders exact/F1 examples from most to least discriminative.
            #
            # Budget example (pop=12, 3x=36 offspring, 30 val examples):
            #   exact/F1: 36×3 → 18, 18×3 → 9, 9×3 → 6, then survivors
            #             receive full-val detailed fitness for comparable selection.
            if deduped_offspring:
                nexus_data = val_data  # Always eval on full val to prevent fitness mismatch
                # SENTINEL PROTOCOL: DISABLED — fold rotation добавляет слишком много variance
                # на BERTScore задачах. Промпты на разных folds несравнимы → UCB получает мусор.
                # Вместо этого используем полный val_data (или NEXUS subset) для всех поколений.
                is_generative = self._is_generative_task(self.dataset_name, self.evaluator)
                # PRISM also broken for f1_macro classification:
                # 3 examples from 4 classes = meaningless metric, kills good prompts
                is_classification = False
                metric = 'exact_match'
                if self.evaluator and hasattr(self.evaluator, 'metrics_evaluator'):
                    metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    is_classification = (metric == 'f1_macro')

                if is_generative or is_classification:
                    # Direct full evaluation for generative and f1_macro tasks — NO PRISM.
                    # PRISM successive halving is proven broken on BERTScore and unsafe
                    # for macro-F1 classification:
                    # - Noisy continuous metric with 10 examples/round = unreliable rankings
                    # - f1_macro cannot be faithfully reduced to 3-example 0/1 batches
                    # - Good offspring killed on noisy partial evaluations
                    # For exact_match tasks: PRISM works great (SQuAD_2 got 1.000).
                    logger.info(
                        f"Gen {generation}: metric={metric} — bypassing PRISM, "
                        f"evaluating all {len(deduped_offspring)} offspring on full val_data"
                    )

                    def _full_eval_offspring(offspring, _data=nexus_data):
                        try:
                            if self.config.get('log_detailed_evaluations', True):
                                eval_details = self.evaluator.evaluate_with_details(
                                    offspring, self.dataset_name, _data
                                )
                                primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(
                                    self.dataset_name
                                )
                                offspring.fitness = eval_details['metrics'][primary_metric]
                            else:
                                offspring.fitness = self.evaluator.evaluate_prompt(
                                    offspring, self.dataset_name, _data, use_cache=False
                                )
                        except Exception as e:
                            logger.error(f"Full eval failed for offspring: {e}")
                            offspring.fitness = 0.0

                    eval_workers = min(4, len(deduped_offspring))
                    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                        list(executor.map(_full_eval_offspring, deduped_offspring))

                    for o in deduped_offspring:
                        o.metadata['_l3_evaluated'] = True

                    all_offspring = deduped_offspring + dedup_skipped

                    logger.info(
                        f"Gen {generation}: Direct eval complete — "
                        f"{len(deduped_offspring)} offspring evaluated on {len(nexus_data)} examples"
                    )

                else:
                    # PRISM successive halving for exact_match/F1 tasks (proven effective)
                    # exact_match/F1: binary/stable metric, fewer examples suffice,
                    # aggressive halving (keep 50%)
                    examples_per_round = min(3, len(nexus_data))
                    elimination_rate = 0.50

                    # Minimum survivors = population_size // 2 (enough for meaningful merge)
                    target_k = max(population_size // 2, 4)

                    # Initialize cumulative tracking + per-example scores for F-Race
                    survivors = list(deduped_offspring)
                    for o in survivors:
                        o.metadata['_prism_cumulative_score'] = 0.0
                        o.metadata['_prism_examples_seen'] = 0
                        o.metadata['_prism_example_scores'] = []  # F-Race: per-example scores

                    example_offset = 0
                    prism_round = 0
                    total_prism_calls = 0
                    racing_eliminated = 0

                    # Use NEXUS critical indices for example ordering if available
                    if hasattr(self, '_nexus_critical_indices') and self._nexus_critical_indices:
                        # Order examples by discriminative power (most discriminative first)
                        ordered_indices = list(self._nexus_critical_indices)
                        # Pad with remaining indices if needed
                        remaining_idx = [i for i in range(len(nexus_data)) if i not in ordered_indices]
                        ordered_indices.extend(remaining_idx)
                        prism_eval_order = [nexus_data[i] for i in ordered_indices[:len(nexus_data)]]
                    else:
                        prism_eval_order = list(nexus_data)

                    # === Successive Halving rounds ===
                    while (len(survivors) > target_k and
                           example_offset + examples_per_round <= len(prism_eval_order)):
                        batch = prism_eval_order[example_offset:example_offset + examples_per_round]
                        batch_size = len(batch)

                        # Evaluate all survivors on this batch (parallel)
                        # F-Race: evaluate per-example for racing elimination
                        def _prism_batch_eval(offspring, _batch=batch, _metric=metric):
                            cache_key = None
                            detailed_cache = getattr(self.evaluator, 'detailed_cache', None)
                            cache_key_builder = getattr(self.evaluator, '_get_cache_key', None)
                            try:
                                # Avoid size-only detailed-cache collisions between different PRISM batches.
                                if isinstance(detailed_cache, dict) and callable(cache_key_builder):
                                    cache_key = cache_key_builder(
                                        offspring.text,
                                        self.dataset_name,
                                        len(_batch),
                                        few_shot_examples=getattr(offspring, 'few_shot_examples', None)
                                    )
                                    detailed_cache.pop(cache_key, None)

                                eval_details = self.evaluator.evaluate_with_details(
                                    offspring, self.dataset_name, _batch
                                )
                                per_example_scores = self._extract_per_example_scores(
                                    _metric, eval_details, _batch
                                )
                            except Exception:
                                per_example_scores = [0.0] * batch_size
                            finally:
                                if cache_key is not None and isinstance(detailed_cache, dict):
                                    detailed_cache.pop(cache_key, None)

                            offspring.metadata['_prism_cumulative_score'] += sum(per_example_scores)
                            offspring.metadata['_prism_examples_seen'] += len(per_example_scores)
                            offspring.metadata['_prism_example_scores'].extend(per_example_scores)

                        eval_workers = min(32, len(survivors))
                        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                            list(executor.map(_prism_batch_eval, survivors))

                        total_prism_calls += len(survivors) * batch_size
                        example_offset += batch_size

                        # Rank by average cumulative score
                        survivors.sort(
                            key=lambda p: (
                                p.metadata['_prism_cumulative_score'] /
                                max(1, p.metadata['_prism_examples_seen'])
                            ),
                            reverse=True
                        )

                        # F-Race: statistical elimination before standard halving
                        # Test each survivor against the current leader using Wilcoxon
                        if len(survivors) > target_k:
                            leader_scores = survivors[0].metadata['_prism_example_scores']
                            racing_survivors = [survivors[0]]  # Leader always survives
                            for s in survivors[1:]:
                                if self._racing_should_eliminate(
                                    s.metadata['_prism_example_scores'], leader_scores, min_samples=6
                                ):
                                    racing_eliminated += 1
                                else:
                                    racing_survivors.append(s)
                            # Only apply racing if we don't drop below target_k
                            if len(racing_survivors) >= target_k:
                                survivors = racing_survivors

                        # Eliminate bottom fraction (standard halving)
                        keep_k = max(target_k, math.ceil(len(survivors) * (1 - elimination_rate)))
                        eliminated = len(survivors) - keep_k
                        survivors = survivors[:keep_k]
                        prism_round += 1

                        logger.info(
                            f"PRISM Round {prism_round}: {keep_k + eliminated} → {keep_k} survivors "
                            f"({eliminated} halved, {racing_eliminated} F-Race eliminated, "
                            f"{example_offset} cumulative examples, "
                            f"{total_prism_calls} API calls so far)"
                        )

                    # === Final round: evaluate survivors on ALL remaining examples ===
                    remaining_data = prism_eval_order[example_offset:]
                    if remaining_data and survivors:
                        def _prism_final_eval(offspring, _remaining=remaining_data):
                            try:
                                if self.config.get('log_detailed_evaluations', True):
                                    # Full eval with details for final survivors
                                    eval_details = self.evaluator.evaluate_with_details(
                                        offspring, self.dataset_name, nexus_data
                                    )
                                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(
                                        self.dataset_name
                                    )
                                    offspring.fitness = eval_details['metrics'][primary_metric]
                                else:
                                    offspring.fitness = self.evaluator.evaluate_prompt(
                                        offspring, self.dataset_name, nexus_data, use_cache=False
                                    )
                            except Exception as e:
                                logger.error(f"PRISM final eval failed: {e}")
                                # Use cumulative estimate as fallback
                                seen = max(1, offspring.metadata.get('_prism_examples_seen', 1))
                                offspring.fitness = offspring.metadata.get('_prism_cumulative_score', 0) / seen

                        eval_workers = min(4, len(survivors))
                        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                            list(executor.map(_prism_final_eval, survivors))

                        total_prism_calls += len(survivors) * len(nexus_data)

                    elif survivors:
                        # All examples consumed in halving rounds — use cumulative as fitness
                        for o in survivors:
                            seen = max(1, o.metadata.get('_prism_examples_seen', 1))
                            o.fitness = o.metadata.get('_prism_cumulative_score', 0) / seen

                    logger.info(
                        f"PRISM complete: {len(deduped_offspring)} → {len(survivors)} survivors, "
                        f"{prism_round} halving rounds, {racing_eliminated} F-Race eliminated, "
                        f"{total_prism_calls} total API calls "
                        "(metric=exact/F1, "
                        f"examples_per_round={examples_per_round}, elim_rate={elimination_rate})"
                    )

                    # Clean up PRISM metadata
                    for o in deduped_offspring:
                        o.metadata.pop('_prism_cumulative_score', None)
                        o.metadata.pop('_prism_examples_seen', None)
                        o.metadata.pop('_prism_example_scores', None)

                    all_offspring = survivors + dedup_skipped
                    for o in survivors:
                        o.metadata['_l3_evaluated'] = True
            else:
                # All offspring were duplicates — use dedup_skipped only
                all_offspring = dedup_skipped

        # ===== Step 4: UCB update with actual fitness =====
        # Only update UCB for fully evaluated offspring (L3), not dedup-skipped
        offspring_operator_info = []
        for offspring in all_offspring:
            # Skip dedup-skipped offspring — they have cached fitness, not fresh evaluation
            if offspring.metadata.get('dedup_skip'):
                offspring.metadata.pop('dedup_skip', None)
                offspring.metadata.pop('used_operator', None)
                offspring.metadata.pop('parent_fitness', None)
                offspring.metadata.pop('_l3_evaluated', None)
                continue
            if 'used_operator' in offspring.metadata and 'parent_fitness' in offspring.metadata:
                operator = offspring.metadata['used_operator']
                parent_fitness = offspring.metadata['parent_fitness']
                improvement = offspring.fitness - parent_fitness

                offspring_operator_info.append((offspring, operator, improvement))

                self.ucb_selector.update(operator, improvement, generation=generation)

                self.history.log_evolution_step(
                    generation=generation,
                    operator_used=operator,
                    parent_ids=[],
                    parent_fitnesses=[parent_fitness],
                    offspring=offspring,
                    temperature=self.current_temperature,
                    top_p=self.current_top_p,
                    diversity_score=0.0,
                    accepted=True,
                    rejection_reason=None,
                    metadata={'fitness_improvement': improvement}
                )

                # Clean metadata
                offspring.metadata.pop('used_operator', None)
                offspring.metadata.pop('parent_fitness', None)
                offspring.metadata.pop('_l3_evaluated', None)

        # Track error-operator associations for instance-aware selection
        if hasattr(self, '_error_operator_history') and self.current_errors:
            for offspring, op, imp in offspring_operator_info:
                if op and self.current_errors:
                    for err in self.current_errors[:5]:
                        pred = str(err.get('prediction', '')).lower()
                        truth = str(err.get('ground_truth', '')).lower()
                        if not pred.strip():
                            self._error_operator_history.append(('empty_output', op, imp))
                        elif len(pred) > len(truth) * 3:
                            self._error_operator_history.append(('too_verbose', op, imp))
                        else:
                            self._error_operator_history.append(('wrong_content', op, imp))
            # Keep only last 200 entries
            self._error_operator_history = self._error_operator_history[-200:]

        # ===== Step 4b: GENESIS — extract lessons from successful offspring =====
        # Non-blocking: lessons are extracted in background threads (1 LLM call each)
        task_desc_for_genesis = self.operators.get_task_description(self.dataset_name)
        import threading

        class _GenesisLLMAdapter:
            def __init__(self, llm_client, model):
                self._llm_client = llm_client
                self._model = model

            def call(self, prompt, system_prompt=None, temperature=0.3, max_tokens=150):
                if system_prompt:
                    return self._llm_client.generate(
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': prompt}
                        ],
                        model=self._model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                return self._llm_client.generate(
                    prompt=prompt,
                    model=self._model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

        genesis_llm = _GenesisLLMAdapter(self.llm_client, self.model)
        for offspring, op_name, improvement in offspring_operator_info:
            if improvement > 0:
                # Find the parent that was used
                parent_text = ""
                if offspring.parent_ids:
                    for p in self.population:
                        if p.id in offspring.parent_ids:
                            parent_text = p.text
                            break
                if not parent_text:
                    # Use population median prompt as proxy parent
                    sorted_by_fitness = sorted(self.population, key=lambda p: p.fitness)
                    mid_idx = len(sorted_by_fitness) // 2
                    parent_text = sorted_by_fitness[mid_idx].text if sorted_by_fitness else ""

                if parent_text:
                    # Background thread for lesson extraction — non-blocking.
                    t = threading.Thread(
                        target=self.genesis.extract_lesson,
                        kwargs={
                            'parent_text': parent_text,
                            'offspring_text': offspring.text,
                            'operator_name': op_name,
                            'fitness_delta': improvement,
                            'generation': generation,
                            'task_description': task_desc_for_genesis,
                            'llm_client': genesis_llm,
                            'temperature': 0.3,
                        },
                        daemon=True
                    )
                    t.start()

        # ===== Step 5: (mu+lambda) merge =====
        # Combine elites + offspring + remaining parents
        combined = list(elites) + all_offspring + sorted_pop[elite_size:]

        # Deduplicate by id (elites may appear in sorted_pop too)
        seen_ids = set()
        unique_combined = []
        for p in combined:
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_combined.append(p)

        # ===== Step 6: Diversity tiebreaker selection (reverted from niche protection) =====
        # v11 analysis: niche protection created 35-55 niches from 36-60 candidates (1 prompt = 1 niche),
        # effectively = plain fitness sort. Diversity collapsed: v11 haiku 0.44→0.21, v10 haiku 0.42→0.47.
        # select_with_diversity_tiebreak maintained diversity better in v10.
        # Dataset-adaptive diversity threshold.
        # CommonGen prompts are semantically similar ("compose a sentence using all concepts"),
        # threshold 0.92 rejects too many → diversity collapse (0.445→0.165).
        # Lower threshold preserves more diverse prompts without hurting other datasets.
        _DIVERSITY_THRESHOLDS = {
            'CommonGen': 0.85,  # Prompts inherently similar, need lower bar
        }
        div_threshold = _DIVERSITY_THRESHOLDS.get(self.dataset_name, 0.92)
        self.population = self.diversity_manager.select_with_diversity_tiebreak(
            unique_combined, population_size, similarity_threshold=div_threshold
        )

        # ===== Step 6b: OPERATOR FORGE — update memories =====
        population_median = sorted([p.fitness for p in self.population])[len(self.population) // 2]
        for p in [p for p in self.population if p.generation == generation]:
            op = p.mutation_type or ''
            if op and p.fitness > population_median:
                self.operator_forge.update(op, p.text, p.fitness, population_median)

        forge_stats = self.operator_forge.get_statistics()
        if any(v > 0 for v in forge_stats.values()):
            logger.info(f"OPERATOR FORGE: memories = {forge_stats}")

        # ===== Step 7: Re-evaluate new best on FULL val_data =====
        # NEXUS evaluates on critical subset only. Best prompt needs accurate full fitness.
        if val_data:
            new_best = max(self.population, key=lambda p: p.fitness)
            if new_best.text != getattr(self, '_last_full_eval_prompt', ''):
                try:
                    full_result = self.evaluator.evaluate_with_details(
                        prompt=new_best, dataset_name=self.dataset_name, data=val_data
                    )
                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(
                        self.dataset_name
                    )
                    new_best.fitness = full_result['metrics'][primary_metric]
                    self._last_full_eval_prompt = new_best.text
                    logger.info(
                        f"Gen {generation}: Full val re-eval of new best: "
                        f"fitness={new_best.fitness:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Full val re-eval failed: {e}")

        # ===== Step 8: Island Migration =====
        self._island_migrate(self.population, generation)

        logger.info(
            f"Gen {generation}: (mu+lambda) merge — {len(all_offspring)} offspring evaluated, "
            f"combined pool={len(unique_combined)}, selected={len(self.population)}"
        )

    # ========== Island Model ==========

    def _island_migrate(self, population: List[Prompt], generation: int) -> None:
        """Миграция лучшего промта между островами каждые 2 генерации.

        Light island model to prevent diversity collapse. Each prompt is assigned
        to one of 3 islands. Every 2 generations, best from island i migrates to
        worst slot in island (i+1) % num_islands.
        """
        num_islands = getattr(self, '_num_islands', 3)
        assignments = getattr(self, '_island_assignments', {})

        # Lazy init: assign prompts to islands on first call
        if not assignments:
            self._num_islands = num_islands
            self._island_assignments = {}
            for i, p in enumerate(population):
                self._island_assignments[id(p)] = i % num_islands
            return

        # Ensure all current population members have assignments
        for i, p in enumerate(population):
            if id(p) not in self._island_assignments:
                self._island_assignments[id(p)] = i % num_islands

        # Only migrate every 2 generations, not on Gen 0
        if generation % 2 != 0 or generation == 0:
            return

        islands = [[] for _ in range(num_islands)]
        for p in population:
            island_id = self._island_assignments.get(id(p), 0)
            islands[island_id].append(p)

        # Migrate: best from island i -> worst slot in island (i+1) % num_islands
        for i in range(num_islands):
            if not islands[i]:
                continue
            next_island = (i + 1) % num_islands
            if not islands[next_island]:
                continue

            best = max(islands[i], key=lambda p: p.fitness)
            worst = min(islands[next_island], key=lambda p: p.fitness)

            if best.fitness > worst.fitness:
                # Copy best prompt to worst slot
                worst.text = best.text
                worst.fitness = 0.0  # Will be re-evaluated next generation
                worst.few_shot_examples = list(best.few_shot_examples) if best.few_shot_examples else []
                self._island_assignments[id(worst)] = next_island
                logger.info(
                    f"ISLAND MIGRATION: Best from island {i} "
                    f"(fitness={best.fitness:.4f}) -> island {next_island}"
                )

        # Clean up stale assignments (prompts no longer in population)
        current_ids = {id(p) for p in population}
        stale_keys = [k for k in self._island_assignments if k not in current_ids]
        for k in stale_keys:
            del self._island_assignments[k]

    # ========== Main Run Method ==========

    def run(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        dev_data: Optional[List[Dict]] = None,
        show_progress: bool = True,
        initial_population: Optional[List[Prompt]] = None,
        start_generation: int = 0
    ) -> Prompt:
        """
        Запуск полного алгоритма RIDER.

        Args:
            train_data: Train данные (для инициализации)
            val_data: Validation данные (для эволюции)
            dev_data: Dev данные (опционально, для финального отбора)
            show_progress: Показывать прогресс
            initial_population: Готовая начальная популяция (для warm start).
                Если передана, пропускаем zero-order initialization и переоцениваем
                промпты на новых train_data.
            start_generation: Поколение, с которого начать (для crash recovery).
                Если > 0, пропускаем инициализацию и ранние поколения.

        Returns:
            Лучший промпт
        """
        logger.info("=" * 70)
        logger.info(f"RIDER - {self.dataset_name}")
        logger.info("=" * 70)
        logger.info(f"Config: pop={self.config['population_size']}, gen={self.config['num_generations']}")
        logger.info(f"Features: pareto={self.use_pareto}, memory=True, reflection=True")

        # Сохраняем val_data для soft restart и diversity injection.
        self._val_data = val_data
        # CHIMERA ENGINE: Сохраняем train_data для few-shot evolution
        self._train_data = train_data

        # SENTINEL PROTOCOL: DISABLED — fold rotation добавляет слишком много variance.
        # Валидировано на v14 CommonGen: UCB получал 100% success / 0.000 improvement
        # потому что промпты на разных folds несравнимы. Best fitness скакал 0.53-0.65
        # в зависимости от fold, а не от качества промпта.
        self._sentinel_folds = None

        # ===== Stage 1: Initialization (or Warm Start or Crash Recovery) =====
        if start_generation > 0 and initial_population:
            # Crash recovery: population already loaded from checkpoint, skip init
            self.population = initial_population[:self.config['population_size']]
            self.best_prompt = copy.deepcopy(max(self.population, key=lambda p: p.fitness))
            logger.info(
                f"Crash recovery: resuming from Gen {start_generation} with "
                f"{len(self.population)} prompts (best={self.best_prompt.fitness:.4f})"
            )
            # Restore GENESIS lessons from checkpoint
            try:
                genesis_pattern = str(self.history.history_dir / "genesis_gen*.json")
                genesis_files = sorted(glob_module.glob(genesis_pattern))
                if genesis_files:
                    with open(genesis_files[-1], 'r', encoding='utf-8') as f:
                        genesis_data = json.load(f)
                    self.genesis = GenesisMemory.from_dict(genesis_data, max_lessons=5)
                    logger.info(f"GENESIS: Restored {self.genesis._total_lessons_extracted} lessons from checkpoint")
            except Exception as e:
                logger.debug(f"GENESIS restore failed (non-critical): {e}")
            if show_progress:
                print(f"\n[Stage 1/3] Crash recovery: resuming from Gen {start_generation}, "
                      f"best={self.best_prompt.fitness:.4f}")
        elif initial_population:
            pop_size = self.config['population_size']
            # Берём лучшие pop_size промптов из warm start популяции
            warm_pop = initial_population[:pop_size]
            if show_progress:
                print(f"\n[Stage 1/3] Warm start: {len(warm_pop)} prompts loaded, re-evaluating on new data...")
            logger.info(f"Warm start: using {len(warm_pop)} prompts from previous run")

            # Переоцениваем на VAL данных (сбрасываем кэш - данные изменились) — PARALLEL
            def _eval_warm(idx_prompt):
                idx, p = idx_prompt
                p.evaluation_cache = {}
                p.fitness = self.evaluator.evaluate_prompt(
                    p, self.dataset_name, val_data, use_cache=False
                )
                return idx

            eval_workers = min(4, len(warm_pop))
            with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                list(executor.map(_eval_warm, enumerate(warm_pop)))

            # Sequential logging
            for prompt in warm_pop:
                self.history.log_evolution_step(
                    generation=0,
                    operator_used="warm_start",
                    parent_ids=[],
                    parent_fitnesses=[],
                    offspring=prompt,
                    temperature=self.temperature,
                    top_p=0.95,
                    diversity_score=0.0,
                    accepted=True,
                    rejection_reason=None,
                    metadata={"warm_start": True}
                )

            self.population = warm_pop
            logger.info(
                "Warm start re-evaluation done: "
                f"avg_fitness={np.mean([p.fitness for p in self.population]):.4f}"
            )
        else:
            if show_progress:
                print("\n[Stage 1/3] Initializing population...")
            self.population = self.initialize_population(train_data, val_data, show_progress)

        # ===== NEXUS PROTOCOL — initial classification =====
        self._nexus_critical_indices = self._nexus_classify_examples(val_data)
        self._nexus_val_data = [val_data[i] for i in self._nexus_critical_indices]
        self._last_full_eval_prompt = ''
        logger.info(
            f"NEXUS PROTOCOL: {len(self._nexus_critical_indices)}/{len(val_data)} "
            f"critical examples identified: {self._nexus_critical_indices}"
        )

        # ===== Stage 2: Evolution =====
        if show_progress:
            print(f"\n[Stage 2/3] Evolution ({self.config['num_generations']} generations)...")

        for generation in range(start_generation, self.config['num_generations']):
            gen_start_time = time.time()

            if show_progress:
                print(f"\nGeneration {generation + 1}/{self.config['num_generations']}")

            # Оцениваем ТОЛЬКО новые промпты (fitness == 0.0).
            # Старые промпты (элиты из предыдущего поколения) сохраняют fitness.
            # Это экономит ~60% API-вызовов и убирает шум от LLM-стохастичности,
            # который маскировал стагнацию (±1-2% колебания fitness при T=0.0).
            new_prompts = [p for p in self.population if p.fitness == 0.0]
            if new_prompts:
                run_eval_results = {}

                def _eval_new_prompt(idx_prompt):
                    idx, p = idx_prompt
                    try:
                        if self.config.get('log_detailed_evaluations', True):
                            ed = self.evaluator.evaluate_with_details(
                                p, self.dataset_name, val_data
                            )
                            pm = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                            p.fitness = ed['metrics'][pm]
                            return idx, ed
                        else:
                            p.fitness = self.evaluator.evaluate_prompt(
                                p, self.dataset_name, val_data, use_cache=True
                            )
                            return idx, None
                    except Exception as e:
                        logger.error(f"Run eval failed for prompt {idx}: {e}")
                        p.fitness = 0.0
                        return idx, None

                eval_workers = min(4, len(new_prompts))
                with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                    for idx, ed in executor.map(_eval_new_prompt, enumerate(new_prompts)):
                        run_eval_results[idx] = ed

                # Sequential logging
                for idx, prompt in enumerate(new_prompts):
                    ed = run_eval_results.get(idx)
                    if ed and self.config.get('log_detailed_evaluations', True):
                        self.history.log_detailed_evaluation(
                            prompt_id=prompt.id,
                            generation=generation,
                            dataset_name=self.dataset_name,
                            evaluation_details=ed
                        )

            # Собираем ошибки лучшего промпта для error-directed evolution
            # P0-3: Skip if best prompt hasn't changed (errors would be identical)
            if self.config.get('log_detailed_evaluations', True):
                current_best = max(self.population, key=lambda p: p.fitness)
                if current_best.text != getattr(self, '_last_error_collection_prompt', '') and self.stagnation_count == 0:
                    self._last_error_collection_prompt = current_best.text
                    try:
                        best_eval = self.evaluator.evaluate_with_details(
                            current_best, self.dataset_name, val_data
                        )
                        self.current_errors = []
                        current_correct = []  # correct examples для CED contrastive pairs
                        preds = best_eval.get('predictions', [])
                        truths = best_eval.get('ground_truth', [])

                        # BUG FIX: metric-aware error classification.
                        metric = 'exact_match'
                        if self.evaluator and hasattr(self.evaluator, 'metrics_evaluator'):
                            metric = self.evaluator.metrics_evaluator.get_primary_metric_name(
                                self.dataset_name
                            )

                        if metric == 'bert_score_f1':
                            # Generative tasks: use per-example metric thresholds, not string equality.
                            # Evaluate each example individually to get per-example scores
                            for i, (pred, truth) in enumerate(zip(preds, truths)):
                                if i >= len(val_data):
                                    break
                                try:
                                    score = self.evaluator.evaluate_prompt(
                                        current_best, self.dataset_name, [val_data[i]], use_cache=False
                                    )
                                except Exception:
                                    score = 0.0

                                if score < 0.4:
                                    self.current_errors.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth,
                                        'score': score
                                    })
                                elif score >= 0.7:
                                    current_correct.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth,
                                        'score': score
                                    })
                        elif metric == 'f1':
                            for i, (pred, truth) in enumerate(zip(preds, truths)):
                                if i >= len(val_data):
                                    break
                                score = self._single_example_f1_score(pred, truth)
                                if score < 0.4:
                                    self.current_errors.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth,
                                        'score': score
                                    })
                                elif score >= 0.7:
                                    current_correct.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth,
                                        'score': score
                                    })
                        elif metric == 'f1_macro':
                            # Classification labels are discrete classes, so exact label match is correct.
                            for i, (pred, truth) in enumerate(zip(preds, truths)):
                                if i >= len(val_data):
                                    break
                                pred_label = str(pred).strip().lower()
                                truth_label = str(truth).strip().lower()
                                if pred_label != truth_label:
                                    self.current_errors.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth
                                    })
                                else:
                                    current_correct.append({
                                        'index': i,
                                        'input': val_data[i],
                                        'prediction': pred,
                                        'ground_truth': truth
                                    })
                        else:
                            # exact_match path intentionally keeps the legacy exact-string logic.
                            for i, (pred, truth) in enumerate(zip(preds, truths)):
                                if str(pred).strip().lower() != str(truth).strip().lower():
                                    if i < len(val_data):
                                        self.current_errors.append({
                                            'index': i,
                                            'input': val_data[i],
                                            'prediction': pred,
                                            'ground_truth': truth
                                        })
                                else:
                                    # Сохраняем успешные примеры для CED
                                    if i < len(val_data):
                                        current_correct.append({
                                            'index': i,
                                            'input': val_data[i],
                                            'prediction': pred,
                                            'ground_truth': truth
                                        })
                        # Сохраняем correct_examples в metadata лучшего промпта
                        # CED использует их для контрастивных пар (SUCCESS vs FAILURE)
                        current_best.metadata['correct_examples'] = current_correct[:5]
                    except Exception as e:
                        logger.debug(f"Error tracking failed: {e}")
                else:
                    logger.debug(f"Gen {generation}: Skipping error collection — best prompt unchanged")

            # update_ucb_statistics() убран из run().
            # UCB уже обновляется в evolve_generation() Step 4 (строки ~984-1009),
            # metadata очищается там же → этот вызов был пустым циклом.

            # NEXUS re-classification every 2 generations
            if generation > 0 and generation % 2 == 0:
                self._nexus_critical_indices = self._nexus_classify_examples(val_data)
                self._nexus_val_data = [val_data[i] for i in self._nexus_critical_indices]
                logger.info(
                    f"NEXUS re-classification: {len(self._nexus_critical_indices)} critical examples"
                )

            # Statistics
            fitnesses = [p.fitness for p in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            diversity_score = self.diversity_manager.compute_diversity_score(self.population)

            # Update fitness history for adaptive diversity stagnation detection
            self.diversity_manager.update_fitness_history(best_fitness)

            # Stagnation detection + adaptive temperature
            self._detect_and_respond_to_stagnation(generation, best_fitness)
            # Передаём текущую температуру операторам
            self.operators.temperature = self.current_temperature

            # Update best
            current_best = max(self.population, key=lambda p: p.fitness)
            if self.best_prompt is None or current_best.fitness > self.best_prompt.fitness:
                self.best_prompt = copy.deepcopy(current_best)

            # Early stopping — perfect fitness
            if self.best_prompt and self.best_prompt.fitness >= 0.999:
                logger.info(
                    f"EARLY STOPPING at Gen {generation} — "
                    f"perfect fitness {self.best_prompt.fitness:.4f}"
                )
                # Log generation summary before breaking (need stats below)
                self._early_stop = True
            else:
                self._early_stop = False

            # Early stopping — no improvement for 5+ gens after Gen 4
            # Was 3 gens — too aggressive. PB finds best prompts at Gen 5-8.
            if generation >= 4 and (self.stagnation_count >= 5 or self._restart_count >= 2):
                logger.info(
                    f"EARLY STOPPING at Gen {generation} — "
                    f"no improvement for {self.stagnation_count} consecutive gens "
                    f"(soft_restarts={self._restart_count})"
                )
                self._early_stop = True

            # ИСПРАВЛЕНО: Log generation summary с правильными параметрами
            # Собрать UCB статистику
            ucb_stats_raw = self.ucb_selector.get_statistics()
            ucb_distribution = {op: stats['uses'] / sum(s['uses'] for s in ucb_stats_raw.values())
                                for op, stats in ucb_stats_raw.items()} if ucb_stats_raw else {}
            ucb_success_rates = {op: stats.get('success_rate', 0.0)
                                 for op, stats in ucb_stats_raw.items()} if ucb_stats_raw else {}

            # Собрать diversity статистику
            div_stats_raw = self.diversity_manager.get_statistics(self.population)
            diversity_stats = {
                'diversity_score': diversity_score,
                'avg_similarity': div_stats_raw.get('avg_similarity', 0.0)
            }

            # Собрать hyperparameter статистику
            hyperparameter_stats = {
                'avg_temperature': self.current_temperature,
                'current_temperature': self.current_temperature,
                'avg_top_p': self.current_top_p,
                'current_top_p': self.current_top_p,
                'stagnation_count': self.stagnation_count,
                'restart_count': self._restart_count,
                'base_temperature': self.base_temperature,
                'offspring_multiplier': getattr(self, '_last_offspring_multiplier', None),
                'num_offspring': getattr(self, '_last_num_offspring', None)
            }

            # Elite IDs (top-3)
            elite = sorted(self.population, key=lambda p: p.fitness, reverse=True)[:self.config['elite_size']]
            elite_ids = [p.id for p in elite]

            # Timing
            gen_time = time.time() - gen_start_time

            # Memory update (каждые 3 поколения)
            memory_updated = False
            memory_insights = None
            if generation > 0 and generation % 3 == 0:
                memory_updated = True
                # get_summary() - правильное имя метода в LongTermMemory
                memory_insights = self.long_term_memory.get_summary() if hasattr(self, 'long_term_memory') else None

            # Снимок API-статистики для анализа сходимости
            api_usage_snapshot = self.llm_client.snapshot_generation(generation)
            api_usage_gen = self.llm_client.get_generation_usage(generation)
            hyperparameter_stats['api_calls_cumulative'] = api_usage_snapshot.get('api_calls', 0)
            hyperparameter_stats['api_calls_this_gen'] = api_usage_gen.get('api_calls', 0)
            hyperparameter_stats['total_tokens_cumulative'] = api_usage_snapshot.get('total_tokens', 0)
            hyperparameter_stats['tokens_this_gen'] = api_usage_gen.get('total_tokens', 0)
            hyperparameter_stats['prompt_tokens_cumulative'] = api_usage_snapshot.get('prompt_tokens', 0)
            hyperparameter_stats['completion_tokens_cumulative'] = api_usage_snapshot.get('completion_tokens', 0)
            hyperparameter_stats['cost_usd_cumulative'] = api_usage_snapshot.get('cost_usd', 0.0)
            hyperparameter_stats['cost_usd_this_gen'] = api_usage_gen.get('cost_usd', 0.0)

            self.history.log_generation_summary(
                generation=generation,
                population=self.population,  # ПРАВИЛЬНО!
                elite_ids=elite_ids,  # ПРАВИЛЬНО!
                ucb_stats={'distribution': ucb_distribution, 'success_rates': ucb_success_rates},
                diversity_stats=diversity_stats,
                hyperparameter_stats=hyperparameter_stats,
                timing=gen_time,
                memory_updated=memory_updated,
                memory_insights=memory_insights,
                eliminated_operators=getattr(self, '_last_eliminated_operators', None),
                bayesian_eliminated=getattr(self, '_last_bayesian_eliminated', None)
            )

            # Save checkpoint with population for crash recovery
            self.history.save_checkpoint(generation, population=self.population)

            # Save GENESIS lessons alongside checkpoint
            if self.genesis and self.genesis._total_lessons_extracted > 0:
                try:
                    genesis_file = self.history.history_dir / f"genesis_gen{generation:03d}.json"
                    with open(genesis_file, 'w', encoding='utf-8') as f:
                        json.dump(self.genesis.to_dict(), f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.debug(f"GENESIS checkpoint save failed: {e}")

            if show_progress:
                print(f"  Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Diversity: {diversity_score:.4f}")
                print(f"  Best prompt: {self.best_prompt.text[:80].encode('ascii', 'replace').decode()}...")

                # Show top operators
                ucb_stats = self.ucb_selector.get_statistics()
                if ucb_stats:
                    top_ops = sorted(
                        ucb_stats.items(),
                        key=lambda x: x[1].get('avg_reward', 0),
                        reverse=True
                    )[:3]
                    print("  Top operators: ", end="")
                    for op, stats in top_ops:
                        print(f"{op}({stats['uses']}, R={stats['avg_reward']:.3f}) ", end="")
                    print()

            # Break after logging if early stop triggered
            if getattr(self, '_early_stop', False):
                if show_progress:
                    print(f"  >>> EARLY STOPPING at Gen {generation}")
                break

            # Evolve (except last generation)
            if generation < self.config['num_generations'] - 1:
                self.evolve_generation(generation, val_data, show_progress=False)

        # ===== Stage 3: Final Selection =====
        if show_progress:
            print("\n[Stage 3/3] Final selection...")

        logger.info("=" * 70)
        logger.info("RIDER Completed")
        logger.info(f"Best fitness: {self.best_prompt.fitness:.4f}")
        logger.info(f"Best prompt: {self.best_prompt.text}")
        logger.info("=" * 70)

        return self.best_prompt

    def select_ensemble(self, k: int = 5) -> List[Prompt]:
        """
        Выбирает ансамбль из k разнообразных промптов.

        Args:
            k: Размер ансамбля

        Returns:
            Список из k промптов
        """
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        candidates = sorted_pop[:min(15, len(sorted_pop))]

        if self.use_pareto:
            ensemble = self.pareto_selection(candidates, k)
        else:
            ensemble = self.kdpp_selector.select_ensemble(candidates, k)

        logger.info(f"Selected ensemble of {len(ensemble)} prompts")

        return ensemble

    def get_statistics(self) -> Dict:
        """
        Получить статистику алгоритма.

        Returns:
            Словарь со статистикой
        """
        return {
            'ucb_statistics': self.ucb_selector.get_statistics(),
            'diversity_manager_stats': {
                'adaptive_enabled': self.diversity_manager.adaptive,
                'min_threshold': self.diversity_manager.min_threshold,
                'max_threshold': self.diversity_manager.max_threshold
            },
            'memory_stats': {
                'success_patterns': len(self.long_term_memory.success_patterns),
                'failure_patterns': len(self.long_term_memory.failure_patterns)
            },
            # API usage tracking
            'api_usage': self.llm_client.get_usage_stats(),
            'population_size': len(self.population),
            'best_fitness': self.best_prompt.fitness if self.best_prompt else 0.0
        }

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            "RIDER("
            f"dataset={self.dataset_name}, "
            f"pop={self.config.get('population_size')}, "
            f"gen={self.config.get('num_generations')}, "
            f"pareto={self.use_pareto})"
        )
