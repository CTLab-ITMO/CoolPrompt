"""
Адаптивное управление гиперпараметрами LLM (температура, top-p).

Этот модуль реализует динамическую адаптацию температуры и top-p
в зависимости от прогресса эволюции и текущего разнообразия популяции.

Ключевая идея:
- Низкое разнообразие → увеличить температуру (exploration)
- Высокое разнообразие → уменьшить температуру (exploitation)
- Прогресс эволюции → постепенное охлаждение (simulated annealing)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterHistory:
    """История изменения гиперпараметров для анализа"""
    generation: int
    temperature: float
    top_p: float
    diversity_score: float
    progress: float
    reason: str  # Причина изменения


class AdaptiveHyperparameters:
    """
    Управление адаптивной температурой и top-p для LLM.

    Адаптирует параметры генерации в зависимости от:
    1. Прогресса эволюции (generation / total_generations)
    2. Текущего разнообразия популяции
    3. Заданных границ

    Args:
        initial_temp: Начальная температура (default: 0.7)
        temp_min: Минимальная температура (default: 0.3)
        temp_max: Максимальная температура (default: 1.2)
        initial_top_p: Начальный top-p (default: 0.95)
        diversity_threshold_low: Порог низкого разнообразия (default: 0.3)
        diversity_threshold_high: Порог высокого разнообразия (default: 0.7)

    Example:
        >>> adapter = AdaptiveHyperparameters(initial_temp=0.7)
        >>> # Получить адаптивные параметры
        >>> temp, top_p = adapter.get_parameters(
        ...     generation=5,
        ...     total_generations=12,
        ...     current_diversity=0.25  # Низкое разнообразие
        ... )
        >>> print(f"Temperature: {temp:.2f}, Top-p: {top_p:.2f}")
        Temperature: 0.91, Top-p: 0.90
        >>> # История изменений
        >>> history = adapter.get_history()
    """

    def __init__(
        self,
        initial_temp: float = 0.7,
        temp_min: float = 0.3,
        temp_max: float = 1.2,
        initial_top_p: float = 0.95,
        diversity_threshold_low: float = 0.3,
        diversity_threshold_high: float = 0.7
    ):
        """Инициализация адаптера гиперпараметров"""
        self.initial_temp = initial_temp
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.initial_top_p = initial_top_p
        self.diversity_threshold_low = diversity_threshold_low
        self.diversity_threshold_high = diversity_threshold_high

        # История изменений для анализа
        self.history: List[HyperparameterHistory] = []

        logger.info(
            f"AdaptiveHyperparameters initialized: "
            f"temp=[{temp_min}, {temp_max}], "
            f"diversity_thresholds=[{diversity_threshold_low}, {diversity_threshold_high}]"
        )

    def get_temperature(
        self,
        generation: int,
        total_generations: int,
        current_diversity: float
    ) -> float:
        """
        Вычислить адаптивную температуру.

        Формула:
        T = base_temp × (1 - 0.3 × progress) × diversity_adjustment

        где:
        - base_temp: начальная температура
        - progress: generation / total_generations
        - diversity_adjustment:
            * diversity < 0.3: 1.3 (повысить T для exploration)
            * diversity > 0.7: 0.8 (понизить T для exploitation)
            * иначе: 1.0 (не менять)

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений
            current_diversity: Текущий diversity score [0, 1]

        Returns:
            Адаптивная температура в пределах [temp_min, temp_max]
        """
        # Прогресс эволюции [0, 1]
        progress = generation / max(total_generations, 1)

        # Базовая температура с охлаждением (simulated annealing)
        base_temp = self.initial_temp * (1 - 0.3 * progress)

        # Адаптация под diversity
        if current_diversity < self.diversity_threshold_low:
            # Низкое разнообразие → увеличить температуру
            diversity_adjustment = 1.3
            reason = "low_diversity_exploration"
        elif current_diversity > self.diversity_threshold_high:
            # Высокое разнообразие → уменьшить температуру
            diversity_adjustment = 0.8
            reason = "high_diversity_exploitation"
        else:
            # Нормальное разнообразие
            diversity_adjustment = 1.0
            reason = "normal_diversity"

        # Итоговая температура
        temperature = base_temp * diversity_adjustment

        # Ограничение в заданных пределах
        temperature = max(self.temp_min, min(self.temp_max, temperature))

        # Логирование
        logger.debug(
            f"Gen {generation}: T={temperature:.3f} "
            f"(base={base_temp:.3f}, div_adj={diversity_adjustment:.2f}, "
            f"diversity={current_diversity:.3f}, reason={reason})"
        )

        return temperature

    def get_top_p(
        self,
        generation: int,
        total_generations: int
    ) -> float:
        """
        Вычислить адаптивный top-p (nucleus sampling).

        Стратегия: постепенное уменьшение top-p для сужения search space.

        Schedule:
        - progress < 0.3: top_p = 0.95 (широкий поиск)
        - progress < 0.7: top_p = 0.90 (умеренный поиск)
        - progress >= 0.7: top_p = 0.85 (узкий поиск)

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений

        Returns:
            Адаптивный top-p
        """
        progress = generation / max(total_generations, 1)

        if progress < 0.3:
            top_p = 0.95
            reason = "early_exploration"
        elif progress < 0.7:
            top_p = 0.90
            reason = "mid_refinement"
        else:
            top_p = 0.85
            reason = "late_exploitation"

        logger.debug(
            f"Gen {generation}: top_p={top_p:.2f} "
            f"(progress={progress:.2f}, reason={reason})"
        )

        return top_p

    def get_parameters(
        self,
        generation: int,
        total_generations: int,
        current_diversity: float
    ) -> Tuple[float, float]:
        """
        Получить адаптивные температуру и top-p одновременно.

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений
            current_diversity: Текущий diversity score

        Returns:
            Tuple[temperature, top_p]
        """
        progress = generation / max(total_generations, 1)

        temperature = self.get_temperature(generation, total_generations, current_diversity)
        top_p = self.get_top_p(generation, total_generations)

        # Определить основную причину адаптации
        if current_diversity < self.diversity_threshold_low:
            reason = "low_diversity_exploration"
        elif current_diversity > self.diversity_threshold_high:
            reason = "high_diversity_exploitation"
        elif progress < 0.3:
            reason = "early_exploration"
        elif progress >= 0.7:
            reason = "late_exploitation"
        else:
            reason = "normal_evolution"

        # Сохранить в истории для анализа
        self.history.append(HyperparameterHistory(
            generation=generation,
            temperature=temperature,
            top_p=top_p,
            diversity_score=current_diversity,
            progress=progress,
            reason=reason
        ))

        return temperature, top_p

    def get_history(self) -> List[Dict]:
        """
        Получить историю изменений гиперпараметров.

        Returns:
            Список словарей с историей по каждому поколению
        """
        return [
            {
                'generation': h.generation,
                'temperature': h.temperature,
                'top_p': h.top_p,
                'diversity_score': h.diversity_score,
                'progress': h.progress,
                'reason': h.reason
            }
            for h in self.history
        ]

    def get_summary_statistics(self) -> Dict:
        """
        Получить сводную статистику адаптации.

        Returns:
            Словарь со статистикой:
            - avg_temperature: средняя температура
            - min/max_temperature: min/max температура
            - avg_top_p: средний top-p
            - reason_distribution: распределение причин адаптации
        """
        if not self.history:
            return {}

        temperatures = [h.temperature for h in self.history]
        top_ps = [h.top_p for h in self.history]
        reasons = [h.reason for h in self.history]

        # Подсчет причин
        reason_counts = {}
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'avg_temperature': sum(temperatures) / len(temperatures),
            'min_temperature': min(temperatures),
            'max_temperature': max(temperatures),
            'avg_top_p': sum(top_ps) / len(top_ps),
            'num_adaptations': len(self.history),
            'reason_distribution': reason_counts
        }

    def reset(self) -> None:
        """Сбросить историю адаптации"""
        self.history = []
        logger.info("Hyperparameter history reset")

    def __repr__(self) -> str:
        """Строковое представление для отладки"""
        return (
            f"AdaptiveHyperparameters("
            f"temp=[{self.temp_min}, {self.temp_max}], "
            f"adaptations={len(self.history)})"
        )
