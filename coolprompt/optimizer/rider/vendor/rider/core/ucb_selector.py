"""
UCB1 (Upper Confidence Bound) алгоритм для адаптивного выбора операторов.

Этот модуль реализует multi-armed bandit подход (UCB1) для динамического
выбора наиболее эффективных эволюционных операторов в процессе оптимизации.

UCB1 балансирует exploitation (использование лучших операторов) и
exploration (тестирование недостаточно изученных операторов).
"""

import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class UCBOperatorSelector:
    """
    UCB1 селектор для адаптивного выбора эволюционных операторов.

    Реализует алгоритм Upper Confidence Bound для multi-armed bandit проблемы,
    где каждый "arm" - это эволюционный оператор.

    Формула UCB1:
        UCB(i) = Q̄ᵢ + c · √(ln N / nᵢ)

    где:
        Q̄ᵢ - средняя награда оператора i
        c - exploration константа (обычно √2 ≈ 1.414)
        N - общее количество использований всех операторов
        nᵢ - количество использований оператора i

    Args:
        operators: Список названий доступных операторов
        c: Exploration константа (default: 1.414 ≈ √2)
        initial_rewards: Начальные награды для task-adaptive инициализации

    Example:
        >>> operators = ['ga_mutation', 'de_mutation', 'reflection_crossover']
        >>> ucb = UCBOperatorSelector(operators, c=1.414)
        >>> # Выбрать оператор
        >>> selected = ucb.select_operator(generation=0, total_generations=10)
        >>> print(selected)
        'ga_mutation'
        >>> # Обновить статистику после применения оператора
        >>> fitness_improvement = 0.05  # Промпт улучшился на 5%
        >>> ucb.update(selected, fitness_improvement)
        >>> # Получить статистику
        >>> stats = ucb.get_statistics()
        >>> print(stats)
    """

    def __init__(
        self,
        operators: List[str],
        c: float = 1.414,
        initial_rewards: Optional[Dict[str, float]] = None,
        use_thompson_sampling: bool = True,
        window_size: int = 5,
        decay_factor: float = 0.9
    ):
        """
        Инициализация UCB/Thompson селектора.

        Args:
            operators: Список названий операторов
            c: Exploration константа (для UCB1)
            initial_rewards: Начальные награды для task-adaptive init
            use_thompson_sampling: Использовать Thompson Sampling вместо UCB1
            window_size: Размер окна для windowed Thompson Sampling (в поколениях)
            decay_factor: Decay для старых наблюдений (0.9 = 10% затухание за поколение)
        """
        self.operators = operators
        self.c = c
        self.use_thompson = use_thompson_sampling

        # Статистика использования (для обоих методов)
        self.counts = {op: 0 for op in operators}
        self.rewards = {op: 0.0 for op in operators}
        self.total_uses = 0

        # История улучшений fitness для анализа
        self.fitness_improvements = defaultdict(list)

        # Thompson Sampling: Beta distributions для каждого оператора
        # Beta(alpha, beta) где alpha = successes + 1, beta = failures + 1
        self.alpha = {op: 1.0 for op in operators}  # successes + prior
        self.beta_param = {op: 1.0 for op in operators}  # failures + prior

        # Windowed Thompson Sampling
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history_window: List[tuple] = []  # (generation, operator, success)
        self.current_generation = 0
        self._initial_rewards = initial_rewards  # Сохраняем для recalculation

        # Epsilon-greedy overlay
        self.epsilon = 0.10            # Базовая вероятность случайного выбора
        self.epsilon_stagnation = 0.25  # При стагнации
        self.stagnation_detected = False

        # PHASE REACTOR v2 — operator selection weight boosts.
        self._phase_boosts: Dict[str, float] = {}

        # Per-generation usage tracking для exploration floor и budget cap.
        # V10-1/V10-2: self.fitness_improvements обновляется ПОСЛЕ генерации всех offspring,
        # поэтому exploration floor и budget cap видели stale данные — все 36 вызовов
        # select_operator() в одном поколении видели одинаковые counts.
        self._gen_usage_counts = defaultdict(lambda: defaultdict(int))

        # Task-adaptive инициализация
        if initial_rewards:
            avg_reward = sum(initial_rewards.values()) / len(initial_rewards) if initial_rewards else 0.0

            for op in self.operators:
                self.counts[op] = 1
                self.rewards[op] = initial_rewards.get(op, avg_reward * 0.5)
                self.total_uses += 1

                if self.use_thompson:
                    op_reward = initial_rewards.get(op, avg_reward * 0.5)
                    pseudo_trials = 5  # Уменьшено с 10 до 5 для быстрой адаптации
                    self.alpha[op] += op_reward * pseudo_trials
                    self.beta_param[op] += (1 - op_reward) * pseudo_trials
        else:
            for op in self.operators:
                self.counts[op] = 1
                self.rewards[op] = 0.0
                self.total_uses += 1

        method_name = "Thompson Sampling (windowed)" if use_thompson_sampling else "UCB1"
        logger.info(
            f"UCBOperatorSelector initialized with {len(operators)} operators, "
            f"method={method_name}, c={c}, window={window_size}, decay={decay_factor}"
        )

    def set_stagnation(self, is_stagnating: bool) -> None:
        """Сигнал от RIDER что эволюция стагнирует."""
        self.stagnation_detected = is_stagnating

    def set_phase_boosts(self, boosts: Dict[str, float]) -> None:
        """PHASE REACTOR v2: temporarily boost operator selection probability.

        Boosts are applied to Thompson Sampling alpha during select_operator().
        A boost of 2.0 doubles the alpha, making the operator ~2x more likely
        to be selected. Operators not in boosts dict get boost=1.0 (no change).
        """
        self._phase_boosts = boosts

    def select_operator(
        self,
        generation: int,
        total_generations: int,
        exclude: Optional[List[str]] = None
    ) -> str:
        """
        Выбрать оператор с использованием UCB1 или Thompson Sampling.
        добавлен epsilon-greedy overlay для forced exploration.
        параметр exclude исключает заблокированные операторы из epsilon-greedy.
        exploration floor (min 3 uses/op in first 2 gens) + budget cap (max 30%).

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений
            exclude: Список операторов, исключённых из selection (task-adaptive filtering).

        Returns:
            Название выбранного оператора
        """
        self.current_generation = generation

        # Список доступных операторов (для epsilon-greedy и Thompson/UCB)
        available = [op for op in self.operators if not exclude or op not in exclude]
        if not available:
            available = self.operators  # Fallback: все операторы если все исключены

        # Exploration floor — per-generation tracking.
        # V9 баг: fitness_improvements обновлялся ПОСЛЕ генерации всех 36 offspring,
        # поэтому все 36 вызовов видели stale данные. XSum Gen 1: eda_mutation 36/36 (100%).
        # Фикс: используем _gen_usage_counts для real-time tracking внутри поколения.
        gen_counts = self._gen_usage_counts[generation]
        if generation <= 1:
            floor_count = 3
            underexplored = [
                op for op in available
                if gen_counts.get(op, 0) < floor_count
            ]
            if underexplored:
                selected = np.random.choice(underexplored)
                gen_counts[selected] += 1
                logger.debug(
                    f"Gen {generation}: EXPLORATION FLOOR → {selected} "
                    f"(gen_uses={gen_counts[selected]})"
                )
                return selected

        # Budget cap — per-generation enforcement.
        # V9 баг: self.counts обновлялся в update() после генерации, поэтому
        # все 36 вызовов видели одинаковый stale ratio. Операторы получали 100% бюджета.
        # Фикс: cap на основе gen_counts (real-time внутри поколения).
        total_gen_uses = sum(gen_counts.values())
        if total_gen_uses > 0:
            max_budget_pct = 0.30  # 30% per generation
            max_allowed = max(3, int(total_gen_uses * max_budget_pct) + 1)
            capped = [op for op in available if gen_counts.get(op, 0) >= max_allowed]
            if capped:
                available = [op for op in available if op not in capped]
                if not available:
                    available = [op for op in self.operators if not exclude or op not in exclude]

        # Epsilon-greedy — принудительный exploration
        # Используем only available операторы — не тратим exploration на blocked
        eps = self.epsilon_stagnation if self.stagnation_detected else self.epsilon
        if np.random.random() < eps:
            selected = np.random.choice(available)
            gen_counts[selected] += 1
            logger.debug(
                f"Gen {generation}: EPSILON exploration → {selected} "
                f"(stagnation={self.stagnation_detected}, excluded={exclude})"
            )
            return selected

        if self.use_thompson:
            # Thompson Sampling: sample from Beta distributions
            # Используем только available операторы (exclude учтён выше)
            # PHASE REACTOR v2 — apply phase boosts to alpha.
            # Boost > 1.0 increases alpha → shifts posterior mean upward → more likely selected.
            samples = {}
            for op in available:
                boost = self._phase_boosts.get(op, 1.0)
                # Sample θ ~ Beta(α * boost, β)
                samples[op] = np.random.beta(self.alpha[op] * boost, self.beta_param[op])

            # Выбрать оператор с максимальным sample
            selected_operator = max(samples, key=samples.get)

            gen_counts[selected_operator] += 1
            logger.debug(
                f"Gen {generation}: Thompson selected {selected_operator} "
                f"(sample={samples[selected_operator]:.3f}, "
                f"α={self.alpha[selected_operator]:.1f}, β={self.beta_param[selected_operator]:.1f})"
            )
        else:
            # UCB1 selection
            # Используем только available операторы (exclude учтён выше)
            ucb_values = {}
            for op in available:
                # Средняя награда
                q_bar = self.rewards[op] / self.counts[op] if self.counts[op] > 0 else 0.0

                # Exploration bonus
                if self.total_uses > 0 and self.counts[op] > 0:
                    exploration = self.c * np.sqrt(np.log(self.total_uses) / self.counts[op])
                else:
                    exploration = 0.0

                # UCB value
                ucb_values[op] = q_bar + exploration

            # Выбрать оператор с максимальным UCB
            selected_operator = max(ucb_values, key=ucb_values.get)

            gen_counts[selected_operator] += 1
            logger.debug(
                f"Gen {generation}: UCB1 selected {selected_operator} "
                f"(UCB={ucb_values[selected_operator]:.3f})"
            )

        return selected_operator

    def update(self, operator: str, fitness_improvement: float, generation: int = None) -> None:
        """
        Обновить статистику оператора после использования.
        Windowed Thompson Sampling с decay.

        Args:
            operator: Название использованного оператора
            fitness_improvement: Улучшение fitness (может быть отрицательным)
            generation: Номер поколения (для windowed TS)
        """
        if operator not in self.operators:
            # chimera and vortex are not UCB-tracked operators — skip silently
            if operator not in ('chimera', 'vortex', 'chimera_swap', 'chimera_crossover'):
                logger.warning(f"Unknown operator: {operator}")
            return

        gen = generation if generation is not None else self.current_generation

        # Нормализовать reward в [0, 1] для UCB1
        reward = max(0.0, min(1.0, 0.5 + fitness_improvement))

        # Обновить статистику (для обоих методов)
        self.counts[operator] += 1
        self.rewards[operator] += reward
        self.total_uses += 1

        # Thompson Sampling: windowed update с decay
        # Сохраняем реальный improvement (не бинарный success/fail)
        # для Extreme Value Credit Assignment.
        if self.use_thompson:
            self.history_window.append((gen, operator, fitness_improvement))
            self._recalculate_thompson_params()

        # Сохранить в истории
        self.fitness_improvements[operator].append(fitness_improvement)

        if self.use_thompson:
            logger.debug(
                f"Updated {operator}: improvement={fitness_improvement:.4f}, "
                f"α={self.alpha[operator]:.1f}, β={self.beta_param[operator]:.1f}"
            )
        else:
            logger.debug(
                f"Updated {operator}: "
                f"improvement={fitness_improvement:.4f}, reward={reward:.3f}, "
                f"total_uses={self.counts[operator]}"
            )

    def _recalculate_thompson_params(self) -> None:
        """
        Пересчитать параметры Thompson Sampling из windowed history с decay.

        Extreme Value Credit Assignment.
        Вместо бинарного success/fail (mean-based) используем MAX fitness improvement
        в скользящем окне для определения success. Оператор считается "успешным"
        если его лучший результат в окне > 0 (а не средний).
        Ref: Fialho et al. 2008, "Extreme Value Based Adaptive Operator Selection" (PPSN).

        Обоснование: high-variance операторы с редкими breakthrough-улучшениями
        наказываются при mean-based credit. С extreme value: один breakthrough (improvement > 0)
        перевешивает множество провалов, что корректно отражает потенциал оператора.
        """
        # Сбросить к priors
        for op in self.operators:
            self.alpha[op] = 1.0
            self.beta_param[op] = 1.0

        # Восстановить initial_rewards pseudo-counts
        if self._initial_rewards:
            avg_reward = sum(self._initial_rewards.values()) / len(self._initial_rewards)
            for op in self.operators:
                op_reward = self._initial_rewards.get(op, avg_reward * 0.5)
                pseudo_trials = 5
                self.alpha[op] += op_reward * pseudo_trials
                self.beta_param[op] += (1 - op_reward) * pseudo_trials

        if not self.history_window:
            return

        # Windowed: учитываем только последние window_size поколений
        current_gen = self.history_window[-1][0]
        min_gen = max(0, current_gen - self.window_size)

        # Extreme Value Credit Assignment.
        # Для каждого оператора в каждом поколении берём MAX improvement.
        # Если max > 0 → success (оператор хотя бы раз произвёл улучшение).
        # Если max <= 0 → failure (все попытки хуже родителя).
        # Ref: Fialho et al. 2008 (PPSN), Li et al. 2014 (IEEE TEVC FRRMAB).
        #
        # history_window entries: (generation, operator, fitness_improvement)
        op_gen_max = defaultdict(lambda: defaultdict(lambda: -float('inf')))

        for gen, op, improvement in self.history_window:
            if gen < min_gen:
                continue
            op_gen_max[op][gen] = max(op_gen_max[op][gen], improvement)

        # Обновляем alpha/beta: один entry per (operator, generation)
        for op in self.operators:
            for gen in sorted(op_gen_max[op].keys()):
                age = current_gen - gen
                weight = self.decay_factor ** age

                # Extreme value: success если max improvement > 0
                if op_gen_max[op][gen] > 0:
                    self.alpha[op] += weight
                else:
                    self.beta_param[op] += weight

        # Negative prior для proven-bad операторов.
        # Если оператор имеет avg_reward < 0.3 и trials > 10 → сильный negative prior.
        # Практически исключает его из selection без полной блокировки.
        for op in self.operators:
            improvements = self.fitness_improvements.get(op, [])
            if len(improvements) >= 10:
                avg_reward = np.mean([max(0, min(1, 0.5 + imp)) for imp in improvements])
                if avg_reward < 0.3:
                    # Установить сильный negative prior
                    self.alpha[op] = max(1.0, self.alpha[op] * 0.3)
                    self.beta_param[op] = max(self.beta_param[op], 10.0)
                    logger.debug(
                        f"Thompson negative prior for {op}: "
                        f"avg_reward={avg_reward:.3f}, trials={len(improvements)}, "
                        f"alpha={self.alpha[op]:.1f}, beta={self.beta_param[op]:.1f}"
                    )

    def get_eliminated_operators(self, min_trials: int = 5, confidence: float = 0.90,
                                  n_samples: int = 10000) -> List[str]:
        """
        Bayesian Automatic Operator Elimination.
        Ослаблена — убран Bonferroni, P>0.90 (было 0.95), min_trials=5 (было 8).

        Вычисляет P(best_operator > operator_i | data) из Beta-постериоров.
        Если P > confidence → оператор элиминируется.

        RIDER Bonferroni делал элиминацию слишком консервативной — операторы
        с 0% success rate на XSum/CommonGen не элиминировались до Gen 6-7.
        Без Bonferroni + P>0.90 + min_trials=5 → элиминация с Gen 2.

        Args:
            min_trials: Минимум использований оператора для элиминации (5).
            confidence: Уровень уверенности (0.90, без поправки Бонферрони).
            n_samples: Количество samples из Beta posterior для Monte Carlo оценки.

        Returns:
            Список операторов, которые следует элиминировать.
        """
        if not self.use_thompson:
            return []

        # Находим операторы с достаточным количеством данных
        candidates = []
        for op in self.operators:
            improvements = self.fitness_improvements.get(op, [])
            if len(improvements) >= min_trials:
                candidates.append(op)

        if len(candidates) < 2:
            return []  # Недостаточно данных для сравнения

        # Без поправки Бонферрони — confidence используется напрямую.
        # Bonferroni делал порог 0.99+ при 8 операторах, что блокировало элиминацию.
        adjusted_threshold = confidence

        # Monte Carlo: сэмплируем из Beta posteriors
        samples = {}
        for op in candidates:
            samples[op] = np.random.beta(self.alpha[op], self.beta_param[op], n_samples)

        # Находим лучший оператор (по среднему posterior)
        best_op = max(candidates, key=lambda op: np.mean(samples[op]))

        # Вычисляем success_rate для защиты от преждевременной элиминации.
        # v8 убил opro_trajectory (60% success, 6 uses) и contrastive_error (50% success)
        # — высоковариантные операторы с реальной ценностью.
        op_success_rates = {}
        for op in candidates:
            improvements = self.fitness_improvements.get(op, [])
            if improvements:
                successes = sum(1 for imp in improvements if imp > 0)
                op_success_rates[op] = successes / len(improvements)
            else:
                op_success_rates[op] = 0.0

        # zero_order защищён от Bayesian elimination.
        # zero_order — лучший инициализатор, но в поздних поколениях имеет 0% success
        # (ожидаемо: промпты с нуля не побеждают оптимизированную популяцию).
        # Elimination блокирует его использование при stagnation escape.
        _BAYESIAN_PROTECTED = {'zero_order'}

        # Для каждого оператора: P(best > op_i)
        eliminated = []
        for op in candidates:
            if op == best_op:
                continue

            if op in _BAYESIAN_PROTECTED:
                logger.debug(f"BAYESIAN PROTECTION: {op} protected — strategic operator")
                continue

            # Снижена защита с 30% до 15%.
            # v10 (0.607, лучший) элиминировал 5 операторов; v13 (0.578) — 0.
            # Причина: 30% порог защищал слабые операторы (20-25% success) от элиминации.
            # 15% защищает только реально ценные операторы.
            if op_success_rates.get(op, 0) > 0.15:
                logger.debug(
                    f"BAYESIAN PROTECTION: {op} protected — "
                    f"success_rate={op_success_rates[op]:.1%} > 30%"
                )
                continue

            p_worse = np.mean(samples[best_op] > samples[op])
            if p_worse > adjusted_threshold:
                eliminated.append(op)
                logger.info(
                    f"BAYESIAN ELIMINATION: {op} eliminated — "
                    f"P({best_op} > {op}) = {p_worse:.4f} > {adjusted_threshold:.4f} "
                    f"(trials={len(self.fitness_improvements[op])}, "
                    f"success_rate={op_success_rates.get(op, 0):.1%})"
                )

        return eliminated

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить подробную статистику использования операторов.

        Returns:
            Словарь со статистикой:
            {
                'operator_name': {
                    'count': <количество использований>,
                    'total_reward': <суммарная награда>,
                    'avg_reward': <средняя награда>,
                    'avg_improvement': <среднее улучшение fitness>,
                    'success_rate': <% случаев с improvement > 0>,
                    # Thompson Sampling stats (если включен):
                    'thompson_mean': <posterior mean = α/(α+β)>,
                    'thompson_alpha': <α параметр>,
                    'thompson_beta': <β параметр>,
                    'thompson_std': <posterior std>,
                },
                ...
            }
        """
        stats = {}

        for op in self.operators:
            count = self.counts[op]
            total_reward = self.rewards[op]

            # Средние значения
            avg_reward = total_reward / count if count > 0 else 0.0

            improvements = self.fitness_improvements.get(op, [])
            avg_improvement = np.mean(improvements) if improvements else 0.0

            # Success rate (% улучшений > 0)
            if improvements:
                success_count = sum(1 for imp in improvements if imp > 0)
                success_rate = success_count / len(improvements)
            else:
                success_rate = 0.0

            stats[op] = {
                'count': count,
                'uses': count,  # Синоним для count (для обратной совместимости)
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'avg_improvement': avg_improvement,
                'success_rate': success_rate,
            }

            # Thompson Sampling: добавить Beta distribution stats
            if self.use_thompson:
                alpha = self.alpha[op]
                beta = self.beta_param[op]
                # Posterior mean: E[θ] = α / (α + β)
                thompson_mean = alpha / (alpha + beta)
                # Posterior std: sqrt(αβ / ((α+β)²(α+β+1)))
                thompson_std = np.sqrt(
                    (alpha * beta) /
                    ((alpha + beta) ** 2 * (alpha + beta + 1))
                )

                stats[op]['thompson_mean'] = thompson_mean
                stats[op]['thompson_alpha'] = alpha
                stats[op]['thompson_beta'] = beta
                stats[op]['thompson_std'] = thompson_std

        return stats

    def get_operator_distribution(self) -> Dict[str, float]:
        """
        Получить распределение использования операторов (в процентах).

        Returns:
            Словарь: {operator_name: percentage}
        """
        if self.total_uses == 0:
            return {op: 0.0 for op in self.operators}

        return {
            op: (self.counts[op] / self.total_uses) * 100
            for op in self.operators
        }

    def reset(self) -> None:
        """Сбросить всю статистику к начальному состоянию."""
        self.counts = {op: 0 for op in self.operators}
        self.rewards = {op: 0.0 for op in self.operators}
        self.total_uses = 0
        self.fitness_improvements = defaultdict(list)
        logger.info("UCB statistics reset")

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            f"UCBOperatorSelector(operators={len(self.operators)}, "
            f"total_uses={self.total_uses}, c={self.c})"
        )


class ContextualLinUCB:
    """
    Contextual LinUCB оператор-селектор.

    Вместо Thompson Sampling (который учит СРЕДНЕЕ качество оператора),
    LinUCB учит КОГДА какой оператор лучше, используя контекст:
    - generation / total_generations (фаза эволюции)
    - population_diversity (разнообразие популяции)
    - stagnation_count / 5 (уровень стагнации)
    - best_fitness (текущий лучший fitness)

    Каждый оператор имеет свою ridge regression модель.
    Выбор: argmax(θᵀx + α√(xᵀA⁻¹x))

    References:
    - Li et al. 2010 "A Contextual-Bandit Approach to
      Personalized News Article Recommendation"
    """

    def __init__(
        self,
        operators: List[str],
        alpha: float = 1.0,
        context_dim: int = 4,
        excluded_operators: Optional[set] = None,
        initial_rewards: Optional[Dict[str, float]] = None
    ):
        """
        Инициализация ContextualLinUCB селектора.

        Args:
            operators: Список названий операторов
            alpha: Exploration параметр для UCB (default: 1.0)
            context_dim: Размерность контекстного вектора (default: 4)
            excluded_operators: Операторы, исключённые из selection
            initial_rewards: Начальные награды для warm start
        """
        self.operators = list(operators)
        self.alpha = alpha
        self.c = alpha  # Drop-in alias с UCBOperatorSelector
        self.context_dim = context_dim
        self.excluded_operators = excluded_operators or set()
        self._initial_rewards = dict(initial_rewards or {})

        # Per-operator ridge regression: A (d×d), b (d×1)
        self._A: Dict[str, np.ndarray] = {}
        self._b: Dict[str, np.ndarray] = {}
        for op in self.operators:
            self._A[op] = np.eye(context_dim)
            self._b[op] = np.zeros(context_dim)
            # Warm start from initial_rewards if provided
            if initial_rewards and op in initial_rewards:
                weight = initial_rewards[op]
                # Add pseudo-observations biased toward operators with higher initial weight
                self._b[op] += weight * np.ones(context_dim) * 2.0

        # Tracking
        self._total_selections = 0
        self._op_selections: Dict[str, int] = {op: 0 for op in operators}
        self._op_rewards: Dict[str, List[float]] = {op: [] for op in operators}
        self._eliminated: set = set()
        self.total_uses = 0
        self.current_generation = 0

        # Epsilon-greedy overlay
        self._epsilon = 0.10
        self._stagnation_epsilon = 0.25
        self._is_stagnating = False
        self._phase_boosts: Dict[str, float] = {}

        # Budget cap: no operator gets more than 30% per generation
        self._gen_usage_counts: Dict[str, int] = {op: 0 for op in operators}
        self._gen_total = 0
        self._budget_cap = 0.30
        self._budget_generation: Optional[int] = None

        # Храним текущий эволюционный контекст для drop-in режима.
        self._context_state = {
            'generation': 0,
            'total_generations': 1,
            'diversity': 0.5,
            'stagnation_count': 0,
            'best_fitness': 0.0,
        }
        self._current_context = self.build_context(
            generation=self._context_state['generation'],
            total_generations=self._context_state['total_generations'],
            diversity=self._context_state['diversity'],
            stagnation_count=self._context_state['stagnation_count'],
            best_fitness=self._context_state['best_fitness']
        )

        logger.info(
            f"ContextualLinUCB initialized with {len(operators)} operators, "
            f"alpha={alpha}, context_dim={context_dim}, "
            f"excluded={self.excluded_operators or 'none'}"
        )

    def build_context(
        self,
        generation: int,
        total_generations: int,
        diversity: float,
        stagnation_count: int,
        best_fitness: float
    ) -> np.ndarray:
        """
        Build 4-dim context vector from current evolutionary state.

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений
            diversity: Разнообразие популяции [0, 1]
            stagnation_count: Счётчик стагнации
            best_fitness: Текущий лучший fitness [0, 1]

        Returns:
            np.ndarray of shape (context_dim,) with values in [0, 1]
        """
        base_context = np.array([
            generation / max(total_generations, 1),   # phase [0, 1]
            diversity,                                 # population diversity [0, 1]
            min(stagnation_count / 5.0, 1.0),          # stagnation level [0, 1]
            best_fitness                               # current best [0, 1]
        ], dtype=float)

        if self.context_dim == len(base_context):
            return base_context
        if self.context_dim < len(base_context):
            return base_context[:self.context_dim]
        return np.pad(base_context, (0, self.context_dim - len(base_context)))

    def set_context(
        self,
        generation: int,
        diversity: float,
        stagnation: int,
        best_fitness: float,
        total_generations: int
    ) -> np.ndarray:
        """
        Сохранить текущий эволюционный контекст для drop-in режима.

        Args:
            generation: Текущее поколение
            diversity: Разнообразие популяции [0, 1]
            stagnation: Счётчик стагнации (или bool-like флаг)
            best_fitness: Текущий лучший fitness [0, 1]
            total_generations: Общее количество поколений

        Returns:
            Сконструированный контекстный вектор.
        """
        self._maybe_rotate_generation(generation)

        self._context_state = {
            'generation': generation,
            'total_generations': total_generations,
            'diversity': diversity,
            'stagnation_count': stagnation,
            'best_fitness': best_fitness,
        }
        self.current_generation = generation
        self._current_context = self.build_context(
            generation=generation,
            total_generations=total_generations,
            diversity=diversity,
            stagnation_count=stagnation,
            best_fitness=best_fitness
        )

        if isinstance(stagnation, bool):
            self._is_stagnating = stagnation
        else:
            self._is_stagnating = stagnation >= 2

        return self._current_context.copy()

    @staticmethod
    def _looks_like_context(value: Any) -> bool:
        """Определить, что позиционный аргумент похож на контекстный вектор."""
        return isinstance(value, (list, tuple, np.ndarray))

    def _normalize_context(self, context: np.ndarray) -> np.ndarray:
        """Преобразовать context к ожидаемой форме (context_dim,)."""
        normalized = np.asarray(context, dtype=float).reshape(-1)
        if normalized.size != self.context_dim:
            raise ValueError(
                f"ContextualLinUCB expected context_dim={self.context_dim}, "
                f"got {normalized.size}"
            )
        return normalized

    def _maybe_rotate_generation(self, generation: Optional[int]) -> None:
        """
        Автоматически сбрасывать per-generation budget tracking при смене поколения.

        Это делает LinUCB drop-in совместимым с rider.py, где reset_generation_counts()
        явно не вызывается, а поколение приходит через select_operator()/update().
        """
        if generation is None:
            return
        if self._budget_generation != generation:
            self.reset_generation_counts()
            self._budget_generation = generation

    def _resolve_context(
        self,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Выбрать либо явный context, либо build_context() из сохранённого состояния."""
        if context is not None:
            self._current_context = self._normalize_context(context)
            return self._current_context.copy()

        if generation is not None or total_generations is not None:
            self.set_context(
                generation=self._context_state['generation'] if generation is None else generation,
                diversity=self._context_state['diversity'],
                stagnation=self._context_state['stagnation_count'],
                best_fitness=self._context_state['best_fitness'],
                total_generations=(
                    self._context_state['total_generations']
                    if total_generations is None else total_generations
                )
            )

        return self._current_context.copy()

    def _resolve_available_operators(
        self,
        available: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_operators: Optional[List[str]] = None
    ) -> List[str]:
        """Собрать итоговый список операторов с учётом исключений и elimination."""
        candidate_ops = list(available) if available is not None else list(self.operators)

        excluded = set(self.excluded_operators)
        excluded.update(self._eliminated)
        if exclude:
            excluded.update(exclude)
        if exclude_operators:
            excluded.update(exclude_operators)

        filtered = [
            op for op in candidate_ops
            if op in self.operators and op not in excluded
        ]
        if filtered:
            return filtered

        # Fallback: как и раньше, если все выбиты elimination'ом — возвращаем хотя бы
        # не globally excluded операторы, чтобы не остаться без выбора.
        fallback_excluded = set(self.excluded_operators)
        if exclude:
            fallback_excluded.update(exclude)
        if exclude_operators:
            fallback_excluded.update(exclude_operators)

        fallback = [
            op for op in self.operators
            if op not in fallback_excluded
        ]
        return fallback or list(self.operators)

    def _apply_budget_cap(self, available: List[str]) -> List[str]:
        """Ограничить операторов, уже выбравших >30% бюджета текущего поколения."""
        if self._gen_total == 0:
            return list(available)

        eligible = [
            op for op in available
            if self._gen_usage_counts.get(op, 0) / max(self._gen_total, 1) < self._budget_cap
        ]
        return eligible or list(available)

    def should_use_epsilon_greedy(self) -> bool:
        """Совместимый helper: использовать ли epsilon-greedy на этом шаге."""
        eps = self._stagnation_epsilon if self._is_stagnating else self._epsilon
        return np.random.random() < eps

    def select_operator(
        self,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        *,
        context: Optional[np.ndarray] = None,
        available: Optional[List[str]] = None,
        available_operators: Optional[List[str]] = None,
        exclude_operators: Optional[List[str]] = None
    ) -> str:
        """
        Select operator using LinUCB with context.

        Формула: score(op) = θᵀx + α√(xᵀA⁻¹x)
        где θ = A⁻¹b — ridge regression estimate для оператора.

        Args:
            generation: Drop-in режим UCBOperatorSelector (текущее поколение)
            total_generations: Drop-in режим UCBOperatorSelector (всего поколений)
            exclude: Drop-in режим UCBOperatorSelector (исключённые операторы)
            context: Явный контекстный вектор (старый LinUCB API)
            available: Явный allow-list операторов (backward compatibility)
            available_operators: Алиас для available
            exclude_operators: Алиас для exclude

        Returns:
            Название выбранного оператора
        """
        if context is None and self._looks_like_context(generation):
            context = generation
            generation = None

        resolved_context = self._resolve_context(
            generation=generation,
            total_generations=total_generations,
            context=context
        )
        candidate_ops = available_operators if available_operators is not None else available
        available_ops = self._resolve_available_operators(
            available=candidate_ops,
            exclude=exclude,
            exclude_operators=exclude_operators
        )
        selectable_ops = self._apply_budget_cap(available_ops)

        # Epsilon-greedy exploration
        if self.should_use_epsilon_greedy():
            selected = selectable_ops[np.random.randint(len(selectable_ops))]
            self._record_selection(selected)
            logger.debug(
                f"ContextualLinUCB EPSILON exploration → {selected} "
                f"(stagnation={self._is_stagnating})"
            )
            return selected

        # LinUCB selection
        best_score = -float('inf')
        best_op = selectable_ops[0]

        for op in selectable_ops:
            A_inv = np.linalg.inv(self._A[op])
            theta = A_inv @ self._b[op]

            # UCB score: θᵀx + α√(xᵀA⁻¹x)
            exploitation = float(theta @ resolved_context)
            phase_boost = self._phase_boosts.get(op, 1.0)
            exploration = (
                self.alpha * phase_boost *
                np.sqrt(float(resolved_context @ A_inv @ resolved_context))
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_op = op

        self._record_selection(best_op)
        logger.debug(
            f"ContextualLinUCB selected {best_op} (score={best_score:.4f})"
        )
        return best_op

    def _record_selection(self, operator: str) -> None:
        """Record operator selection for tracking and budget cap."""
        self._total_selections += 1
        self._op_selections[operator] = self._op_selections.get(operator, 0) + 1
        self._gen_usage_counts[operator] = self._gen_usage_counts.get(operator, 0) + 1
        self._gen_total += 1

    def update(
        self,
        operator: str,
        reward: float,
        generation: Optional[int] = None,
        *,
        context: Optional[np.ndarray] = None
    ) -> None:
        """
        Update operator model with observed reward.

        Ridge regression update: A += xxᵀ, b += r·x

        Args:
            operator: Название использованного оператора
            reward: Наблюдённая награда (fitness improvement)
            generation: Drop-in режим UCBOperatorSelector (текущее поколение)
            context: Контекстный вектор, использованный при выборе
        """
        if context is None and self._looks_like_context(generation):
            context = generation
            generation = None

        if operator not in self._A:
            # chimera and vortex are not tracked — skip silently
            if operator not in ('chimera', 'vortex', 'chimera_swap', 'chimera_crossover'):
                logger.warning(f"ContextualLinUCB: unknown operator {operator}")
            return

        resolved_context = self._resolve_context(generation=generation, context=context)

        # Ridge regression update: A += xxᵀ, b += r·x
        self._A[operator] += np.outer(resolved_context, resolved_context)
        self._b[operator] += reward * resolved_context
        self._op_rewards.setdefault(operator, []).append(reward)
        self.total_uses += 1

        logger.debug(
            f"ContextualLinUCB updated {operator}: reward={reward:.4f}"
        )

    def reset_generation_counts(self) -> None:
        """Reset per-generation budget tracking."""
        self._gen_usage_counts = {op: 0 for op in self.operators}
        self._gen_total = 0

    def set_stagnation(self, is_stagnating: bool) -> None:
        """Update stagnation state for epsilon-greedy."""
        self._is_stagnating = is_stagnating

    def set_phase_boosts(self, boosts: Dict[str, float]) -> None:
        """
        Совместимость с UCBOperatorSelector.

        LinUCB не использует phase boosts как основной механизм, но может мягко
        масштабировать exploration bonus для заданных операторов.
        """
        self._phase_boosts = boosts or {}

    def get_eliminated_operators(
        self,
        min_trials: int = 5,
        confidence: float = 0.85,
        n_samples: int = 10000
    ) -> List[str]:
        """
        Eliminate operators that are consistently worse.
        Uses simple reward averaging (no Monte Carlo needed for LinUCB).

        Args:
            min_trials: Минимум использований оператора для элиминации
            confidence: Не используется напрямую (для совместимости интерфейса),
                        порог определяется как best_avg * 0.3
            n_samples: Не используется напрямую (drop-in совместимость с UCB API)

        Returns:
            List операторов для элиминации
        """
        _ = n_samples  # Параметр сохранён для drop-in совместимости.
        if self._total_selections < min_trials * len(self.operators):
            return []

        avg_rewards: Dict[str, float] = {}
        for op in self.operators:
            rewards = self._op_rewards.get(op, [])
            if len(rewards) >= min_trials:
                avg_rewards[op] = sum(rewards) / len(rewards)

        if len(avg_rewards) < 2:
            return []

        best_avg = max(avg_rewards.values())
        eliminated: List[str] = []

        for op, avg in avg_rewards.items():
            if op == 'zero_order':  # Always protected
                continue
            if avg < best_avg * 0.3 and len(self._op_rewards.get(op, [])) >= min_trials * 2:
                eliminated.append(op)
                logger.info(
                    f"ContextualLinUCB ELIMINATION: {op} — "
                    f"avg_reward={avg:.4f} < {best_avg * 0.3:.4f} "
                    f"(trials={len(self._op_rewards[op])})"
                )

        self._eliminated = set(eliminated)
        return eliminated

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return operator selection and performance statistics.

        Returns:
            Словарь со статистикой по каждому оператору
        """
        stats: Dict[str, Any] = {}
        for op in self.operators:
            rewards = self._op_rewards.get(op, [])
            count = len(rewards)
            total_reward = float(sum(rewards))
            avg_reward = total_reward / count if count else 0.0
            success_rate = (
                sum(1 for reward in rewards if reward > 0) / count
                if count else 0.0
            )
            stats[op] = {
                'count': count,
                'uses': count,
                'total_reward': total_reward,
                'avg_improvement': avg_reward,
                'success_rate': success_rate,
                'selections': self._op_selections.get(op, 0),
                'avg_reward': avg_reward,
                'total_rewards': len(rewards)
            }
        return stats

    def get_operator_distribution(self) -> Dict[str, float]:
        """Совместимость с UCBOperatorSelector: распределение использований операторов."""
        if self.total_uses == 0:
            return {op: 0.0 for op in self.operators}

        return {
            op: (len(self._op_rewards.get(op, [])) / self.total_uses) * 100
            for op in self.operators
        }

    def reset(self) -> None:
        """Сбросить статистику и ridge-модели к начальному состоянию."""
        self._A = {op: np.eye(self.context_dim) for op in self.operators}
        self._b = {op: np.zeros(self.context_dim) for op in self.operators}
        for op, weight in self._initial_rewards.items():
            if op in self._b:
                self._b[op] += weight * np.ones(self.context_dim) * 2.0

        self._total_selections = 0
        self._op_selections = {op: 0 for op in self.operators}
        self._op_rewards = {op: [] for op in self.operators}
        self._eliminated = set()
        self.total_uses = 0
        self._phase_boosts = {}
        self._is_stagnating = False
        self._budget_generation = None
        self.reset_generation_counts()
        self._context_state = {
            'generation': 0,
            'total_generations': 1,
            'diversity': 0.5,
            'stagnation_count': 0,
            'best_fitness': 0.0,
        }
        self._current_context = self.build_context(
            generation=self._context_state['generation'],
            total_generations=self._context_state['total_generations'],
            diversity=self._context_state['diversity'],
            stagnation_count=self._context_state['stagnation_count'],
            best_fitness=self._context_state['best_fitness']
        )
        logger.info("ContextualLinUCB statistics reset")

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            f"ContextualLinUCB(operators={len(self.operators)}, "
            f"alpha={self.alpha}, context_dim={self.context_dim}, "
            f"total_selections={self._total_selections})"
        )
