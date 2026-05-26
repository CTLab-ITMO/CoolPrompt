"""
Счетчик вызовов LLM API для отслеживания использования и затрат.

Этот модуль предоставляет простой счетчик для отслеживания количества
вызовов к различным LLM моделям в процессе эксперимента.
"""

from typing import Dict
from collections import defaultdict


class LLMCallCounter:
    """
    Счетчик вызовов LLM API.

    Отслеживает общее количество вызовов и количество вызовов для каждой модели.
    Полезно для анализа затрат и эффективности различных алгоритмов.

    Example:
        >>> counter = LLMCallCounter()
        >>> counter.increment("gpt-4")
        >>> counter.increment("gpt-4")
        >>> counter.increment("gpt-3.5-turbo")
        >>> stats = counter.get_stats()
        >>> print(stats)
        {'total': 3, 'by_model': {'gpt-4': 2, 'gpt-3.5-turbo': 1}}
    """

    def __init__(self):
        """Инициализация счетчика с нулевыми значениями."""
        self.total_calls: int = 0
        self.calls_by_model: Dict[str, int] = defaultdict(int)

    def increment(self, model: str = "default") -> None:
        """
        Увеличить счетчик для указанной модели.

        Args:
            model: Имя модели (default: "default")
        """
        self.total_calls += 1
        self.calls_by_model[model] += 1

    def reset(self) -> None:
        """Сбросить все счетчики к нулю."""
        self.total_calls = 0
        self.calls_by_model = defaultdict(int)

    def get_stats(self) -> Dict[str, any]:
        """
        Получить статистику вызовов.

        Returns:
            Словарь со структурой:
            {
                'total': <общее количество вызовов>,
                'by_model': {
                    '<model1>': <количество вызовов>,
                    '<model2>': <количество вызовов>,
                    ...
                }
            }
        """
        return {
            'total': self.total_calls,
            'by_model': dict(self.calls_by_model)
        }

    def __repr__(self) -> str:
        """Строковое представление счетчика для отладки."""
        return f"LLMCallCounter(total={self.total_calls}, models={len(self.calls_by_model)})"
