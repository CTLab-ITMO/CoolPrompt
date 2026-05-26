"""
Базовые структуры данных для работы с промптами и результатами экспериментов.

Этот модуль содержит dataclasses для представления промптов и результатов экспериментов.
Эти классы не имеют внешних зависимостей (кроме стандартной библиотеки и numpy).
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class Prompt:
    """
    Класс для представления промпта с метаданными и историей эволюции.

    Attributes:
        text: Текст промпта
        fitness: Значение функции приспособленности (метрика качества)
        generation: Номер поколения, в котором был создан промпт
        parent_ids: Список ID родительских промптов (для отслеживания происхождения)
        mutation_type: Тип мутации/оператора, который создал этот промпт
        evaluation_cache: Кэш результатов оценки на различных примерах
        diversity_features: Эмбеддинги для вычисления diversity (numpy array)
        id: Уникальный идентификатор промпта
        metadata: Дополнительные метаданные (гибкое поле)
    """

    text: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: str = ""
    evaluation_cache: Dict = field(default_factory=dict)
    diversity_features: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: f"p_{random.randint(10000, 99999)}")
    metadata: Dict = field(default_factory=dict)
    few_shot_examples: List[Dict] = field(default_factory=list)  # CHIMERA: evolved examples

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь для сериализации.

        Returns:
            Словарь с полями промпта (embeddings преобразуются в список)
        """
        result = asdict(self)
        if self.diversity_features is not None:
            result['diversity_features'] = self.diversity_features.tolist()
        return result

    def __repr__(self) -> str:
        """Краткое представление промпта для отладки"""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (f"Prompt(id={self.id}, fitness={self.fitness:.3f}, "
                f"gen={self.generation}, text='{text_preview}')")


@dataclass
class ExperimentResult:
    """
    Класс для хранения результатов полного эксперимента.

    Содержит всю информацию о выполненном эксперименте: конфигурацию,
    результаты на test set, историю оптимизации и метаданные.

    Attributes:
        algorithm: Название алгоритма оптимизации (RIDER, EvoPrompt-GA, и т.д.)
        dataset: Название датасета (GSM8K, AG_News, и т.д.)
        model: Имя LLM модели (для отображения)
        temperature: Температура генерации промптов
        run_id: Номер запуска (для повторных экспериментов)
        best_prompt: Лучший найденный промпт (текст)
        test_accuracy: Точность на test set (основная метрика)
        test_metrics: Дополнительные метрики на test set
        optimization_history: История эволюции по поколениям
        total_time: Общее время выполнения эксперимента (секунды)
        num_llm_calls: Количество вызовов LLM API
        config: Конфигурация эксперимента
        timestamp: Временная метка начала эксперимента
    """

    algorithm: str
    dataset: str
    model: str
    temperature: float
    run_id: int

    # Результаты
    best_prompt: str
    test_accuracy: float
    test_metrics: Dict[str, Any]

    # История оптимизации
    optimization_history: List[Dict[str, Any]]

    # Временные метрики
    total_time: float
    num_llm_calls: int

    # Метаданные
    config: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь для сериализации в JSON/CSV.

        Returns:
            Словарь со всеми полями результата
        """
        return asdict(self)

    @property
    def exp_id(self) -> str:
        """
        Уникальный идентификатор эксперимента.

        Формат: <algorithm>_<dataset>_<model>_T<temperature>_R<run_id>
        Пример: RIDER_GSM8K_deepseek-r1_T0.7_R1
        """
        return f"{self.algorithm}_{self.dataset}_{self.model}_T{self.temperature}_R{self.run_id}"

    def __repr__(self) -> str:
        """Краткое представление результата для отладки"""
        return (f"ExperimentResult(exp_id={self.exp_id}, "
                f"test_accuracy={self.test_accuracy:.3f}, "
                f"time={self.total_time:.1f}s)")
